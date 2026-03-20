import asyncio
import logging
import os
import traceback
from datetime import datetime, timedelta
import time
import httpx
import numpy as np
import pandas as pd
from assume.common.market_objects import ClearingMessage

from assume.world import World
from assume.markets.base_market import MarketRole
from assume.common import MarketConfig, WriteOutput
from assume.common.utils import datetime2timestamp, timestamp2datetime
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from assume.markets.clearing_algorithms import clearing_mechanisms
from assume.strategies.learning_strategies import RLStrategy, RenewableRLStrategy
from mango import activate, agent_composed_of, addr, create_acl, Performatives
from mango.util.termination_detection import tasks_complete_or_sleeping
from sqlalchemy import text, create_engine
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PowerWorld(World):
    """
    Extended World class for PowerSUME simulation integrating reinforcement learning agents.
    
    Manages coordination between ASSUME market simulation, Java API for power trading,
    and RL-based bidding strategies. Handles forecast distribution, bid accumulation,
    orderbook processing, and training state management.
    """
    def __init__(
            self,
            java_api_url: str,
            addr: tuple[str, int] | str = "world",
            database_uri: str = "",
            export_csv_path: str = "",
            log_level: str = "INFO",
            distributed_role: bool | None = None,
            test_mode: bool = False,
            session_id: str | None = None,
            forecast_update_frequency: int = 14,
    ) -> None:
        
        """
        Initialize PowerWorld simulation environment.
        
        Args:
            java_api_url (str): Base URL for Java API
            addr (tuple[str, int] | str): Network address for world agent
            database_uri (str): Database connection string (optional)
            export_csv_path (str): Path for CSV export (optional)
            log_level (str): Logging level (default: "INFO")
            distributed_role (bool | None): Whether using distributed architecture
            test_mode (bool): Enable test mode (default: False)
            session_id (str | None): Session identifier
            forecast_update_frequency (int): Forecast update cadence in days (default: 14)
        
        Returns:
            None
        """
        
        super().__init__(
            addr=addr,
            database_uri=database_uri,
            export_csv_path=export_csv_path,
            log_level=log_level,
            distributed_role=distributed_role,
        )

        self.session_id = session_id
        self.test_mode = test_mode
        self.java_api_url = java_api_url
        self.api_endpoints = {
            "health": f"{java_api_url}/health",
            "submit_bids": f"{java_api_url}/submit_rl_bids",
        }

        self.current_episode = 0
        self.eval_completed = False
        self.current_episode_reward = 0.0
        self.current_year = None

        self.training_state = {"days_processed": 0, "total_rewards": 0, "best_reward": float("-inf")}
        self.bidding_strategies = {
            "naive_eom": NaiveSingleBidStrategy,
            "pp_learning": RLStrategy,
            "renewable_eom_learning": RenewableRLStrategy,
        }

        self.forecast_update_frequency = int(max(1, forecast_update_frequency))
        self._forecast_anchor_day_ordinal = None
        self._last_forecast_applied_date = None  # "YYYY-MM-DD"

        # Clearing mechanism override
        clearing_mechanisms["dummy_clearing"] = self.create_dummy_clearing_role
        clearing_mechanisms["pay_as_clear"] = self.create_dummy_clearing_role

        self.accumulated_bids = []
        self.hours_accumulated = 0
        self.waiting_for_response = False
        self.waiting_for_next_day = False
        self.final_cycle_in_progress = False

        # --------------------------
        # Forecast queue + worker
        # --------------------------
        # Async queue lives in the event loop thread; endpoints feed items via call_soon_threadsafe
        self._forecast_queue: asyncio.Queue | None = None
        self._forecast_worker_task: asyncio.Task | None = None

        # NEW: Pre-start buffer for forecast batches arriving before loop is running
        # Items are dicts: {"batch": list[dict], "done_evt": threading.Event | None}
        self._prestart_forecast_batches: list[dict] = []

    # -------------------------- Clearing role --------------------------
    def create_dummy_clearing_role(self, market_config: MarketConfig) -> MarketRole:
        """
        Method to create simulation clearing mechanism. Creates a custom clearing role that integrates with simulation API
        instead of default ASSUME clearing mechanism.
        
        Args:
            market_config (MarketConfig): Market configuration
        
        Returns:
            MarketRole: Custom SimulationClearingRole instance
        """
        
        java_api_url = self.java_api_url
        test_mode = self.test_mode
        session_id = self.session_id

        class SimulationClearingRole(MarketRole):
            def __init__(self, cfg: MarketConfig):
                super().__init__(cfg)
                self.java_api_url = java_api_url
                self.test_mode = test_mode
                self.session_id = session_id

            def clear(self, orderbook, market_products):
                return [], [], [], {}

            async def clear_market(self, market_products: list):
                self.all_orders = []
                return [], []

            def reset(self):
                self.all_orders = []
                super().reset()

        return SimulationClearingRole(market_config)

    # -------------------------- Forecast worker (Queue consumer) --------------------------
    async def _forecast_worker(self):
        """
        Asynchronous worker consuming forecast batches from queue.
        Consumes forecast batches from self._forecast_queue and applies them sequentially.
        Each queue item is a dict: {"batch": list[dict], "done_evt": threading.Event | None}
                
        Returns:
            None
        
        Raises:
            Exceptions are logged; worker gracefully handles cancellation
        """
        try:
            while True:
                item = await self._forecast_queue.get()
                # Poison pill to exit the worker
                if item is None:
                    break

                batch = item.get("batch", [])
                done_evt = item.get("done_evt", None)

                try:
                    written = await self.apply_forecast_from_batch(batch)
                    if hasattr(self, "event_mgr"):
                        self.event_mgr.forecast_applied_event.set()
                    logger.info("Forecast worker applied %d points", int(written))
                except Exception as e:
                    logger.exception("Forecast worker failed applying batch: %s", e)
                finally:
                    if done_evt is not None:
                        try:
                            done_evt.set()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            logger.debug("Forecast worker cancelled")
        except Exception as e:
            logger.exception("Forecast worker crashed: %s", e)

    # -------------------------- Forecast ingest --------------------------
    async def apply_forecast_from_batch(self, batch_units: list[dict]) -> int:
        """
        Ingest fast-array format forecasts into forecaster.
        
        Accepts optimized batch format containing timestamps and arrays for
        residual load, prices, availability, and variable costs per unit.
        Updates _last_forecast_applied_date for midnight-gate logic.
        
        Args:
            batch_units (list[dict] {"unit_ids":[...], "start_epoch":int, "step_sec":int, "len":int, "rl":[...], "av":[...], "vc":[...], "p":[...]}): Batch items with keys:
                - unit_ids: List of unit IDs
                - start_epoch: Unix timestamp for first value
                - step_sec: Seconds between values
                - len: Number of values
                - rl, av, vc, p: Data arrays for residual load, availability, variable costs, price
        
        Returns:
            int: Total forecast points written
        """
        
        if not hasattr(self, "forecaster"):
            return 0
        f = self.forecaster
        written = 0

        for u in batch_units:
            start_epoch = int(u["start_epoch"])
            step_sec = int(u["step_sec"])
            length = int(u["len"])
            rl = u["rl"]
            av = u["av"]
            vc = u["vc"]
            p = u["p"]

            ts = pd.to_datetime([start_epoch + i * step_sec for i in range(length)], unit="s", utc=True).tz_convert(None)
            f.upsert_points("residual_load_EOM", ts, rl)
            f.upsert_points("price_EOM", ts, p)

            for uid in u["unit_ids"]:
                f.upsert_points(f"availability_{uid}", ts, av)
                f.upsert_points(f"var_costs_{uid}", ts, vc)

            written += length * (2 + 2 * len(u["unit_ids"]))

        self._last_forecast_applied_date = str(pd.to_datetime(batch_units[0]["start_epoch"], unit="s").normalize().date())
        logger.info("Forecast written: %d points (anchor day=%s)", written, self._last_forecast_applied_date)
        return written

    # -------------------------- Forecast ingest (store -> forecaster) --------------------------
    async def apply_forecast_from_store(self) -> int:
        """
        Transfer forecasts from store into forecaster.
        
        Extracts snapshot from thread-safe _forecast_store, transfers to
        forecaster, then clears store. Thread-safe via _forecast_lock.
        Sets forecast_applied_event and clears forecast_updated_event.
        
        Returns:
            int: Number of forecast points written
        """
        
        if not hasattr(self, "event_mgr"):
            return 0

        with self.event_mgr._forecast_lock:
            store_snapshot = {u: dict(ts_map) for u, ts_map in self.event_mgr._forecast_store.items()}
            self.event_mgr._forecast_store.clear()

        if hasattr(self, "forecaster") and hasattr(self.forecaster, "upsert_from_event_store"):
            written = self.forecaster.upsert_from_event_store(store_snapshot)
        else:
            written = 0

        day_str = None
        try:
            for _u, by_ts in store_snapshot.items():
                if by_ts:
                    any_ts = next(iter(by_ts.keys()))
                    day_str = str(pd.Timestamp(any_ts).normalize().date())
                    break
        except Exception:
            pass
        if day_str is None:
            day_str = str(pd.Timestamp.fromtimestamp(self.clock.time).normalize().date())

        self._last_forecast_applied_date = day_str
        self.event_mgr.forecast_applied_event.set()
        self.event_mgr.forecast_updated_event.clear()
        logger.info("Forecast applied from store: %s", day_str)
        return written

    # -------------------------- Market registration override --------------------------
    def add_market(self, market_operator_id: str, market_config: MarketConfig) -> None:
        """
        Register market with market simulation clearing override.
        
        Adds a market in ASSUME which uses "dummy_clearing" mechanism via
        create_dummy_clearing_role() before registration.
        
        Args:
            market_operator_id (str): Market operator identifier
            market_config (MarketConfig): Market configuration
        
        Returns:
            None
        """
        
        market_config.market_mechanism = "dummy_clearing"
        super().add_market(market_operator_id, market_config)

    # -------------------------- Run wrappers --------------------------
    def run(self):
        """
        Execute simulation synchronously.
        
        Wrapper converting and executing async_run() from ASSUME via
        asyncio event loop. Handles KeyboardInterrupt gracefully.
        
        Note: One run here corresponds to one episode in RL training.
        
        Returns:
            list: Simulation results from async_run()
        """
        
        start_ts = datetime2timestamp(self.start) if hasattr(self, "start") else 0
        end_ts = datetime2timestamp(self.end) if hasattr(self, "end") else float("inf")
        try:
            return self.loop.run_until_complete(self.async_run(start_ts=start_ts, end_ts=end_ts))
        except KeyboardInterrupt:
            pass

    # -------------------------- Forecast cadence helper --------------------------
    def _should_update_forecast_today(self, d: datetime.date) -> bool:
        """
        Check if forecast should update on given day.
        
        Uses ordinal-based cadence: returns True if days since anchor
        is multiple of forecast_update_frequency.
        
        Args:
            d (datetime.date): Date to check
        
        Returns:
            bool: True if forecast update due today
        """
        ord_day = d.toordinal()
        if self._forecast_anchor_day_ordinal is None:
            self._forecast_anchor_day_ordinal = ord_day
            return True
        return ((ord_day - self._forecast_anchor_day_ordinal) % self.forecast_update_frequency) == 0

    # -------------------------- One step with lightweight midnight sync --------------------------
    async def perform_single_step(self, current_ts):
        """
        Execute single simulation timestep (1 clearing = 24 h) with RL agent integration.
        
        Handles year transitions, forecast synchronization at midnight,
        observation preparation, and bid generation. Returns results dict.
        
        Args:
            current_ts (int): Current timestamp (seconds since epoch)
        
        Returns:
            dict: Step result with keys:
                - timestamp: Current timestamp
                - next_timestamp: Next timestep
                - bids: List of generated bids
                - rewards: Reward dictionary (empty)
        """
        
        tdt = timestamp2datetime(current_ts)

        if self.current_year is None:
            self.current_year = tdt.year
        elif tdt.year != self.current_year:
            if hasattr(self, "update_max_power_for_year"):
                self.update_max_power_for_year(tdt.year)
            self.current_year = tdt.year

        if self.waiting_for_response:
            while self.waiting_for_response:
                await asyncio.sleep(0.1)

        if tdt.hour == 0 and hasattr(self, "event_mgr") and "Operator-RL" in self.unit_operators:
            current_day = tdt.date()
            if self._should_update_forecast_today(current_day):
                if self._last_forecast_applied_date != current_day.isoformat():
                    if not self.event_mgr.forecast_applied_event.is_set():
                        waited, step, max_wait = 0.0, 0.05, 3.0
                        while waited < max_wait and not self.event_mgr.forecast_applied_event.is_set():
                            await asyncio.sleep(step)
                            waited += step

                    if self.event_mgr.forecast_applied_event.is_set():
                        self._last_forecast_applied_date = current_day.isoformat()
                        logger.info("Forecast applied for day %s", current_day.isoformat())
                    else:
                        logger.warning("No forecast for %s — continue with previous values.", current_day.isoformat())

                logger.info(": starting prepare_observations for all RL units")
                for _, unit in self.unit_operators["Operator-RL"].units.items():
                    strat = unit.bidding_strategies.get("EOM")
                    if strat is not None:
                        try:
                            strat.prepare_observations(unit, "EOM")
                        except Exception as e:
                            logger.error("prepare_observations failed for unit %s: %s", getattr(unit, "unit_id", "?"), e)

                self.event_mgr.forecast_applied_event.clear()
                logger.info(": finished prepare_observations for all RL units")

        # --- Generate bids ---
        bids = await self.get_rl_bids()
        next_ts = current_ts + (self.time_step_seconds if hasattr(self, "time_step_seconds") else 3600)
        self.clock.set_time(next_ts)
        return {"timestamp": current_ts, "next_timestamp": next_ts, "bids": bids, "rewards": {}}

    # -------------------------- Orchestrator --------------------------
    async def async_run(self, start_ts, end_ts):
        """
        Main async orchestrator for simulation execution.
        
        Initializes forecast worker, processes timesteps until end_ts or
        eval_completed flag. Handles pre-start forecast buffer drain,
        daily bid accumulation, final cycle completion, and cleanup.
        
        Args:
            start_ts (int): Start timestamp (seconds since epoch)
            end_ts (int): End timestamp (seconds since epoch)
        
        Returns:
            list: All timestep results
        """
        self.end_ts = end_ts
        async with activate(self.container) as container:
            await tasks_complete_or_sleeping(container)

            self._forecast_queue = asyncio.Queue()
            self._forecast_worker_task = asyncio.create_task(self._forecast_worker())

            if self._prestart_forecast_batches:
                try:
                    for item in self._prestart_forecast_batches:
                        await self._forecast_queue.put(item)
                    logger.info("Drained %d pre-start forecast batches into queue", len(self._prestart_forecast_batches))
                finally:
                    self._prestart_forecast_batches.clear()

            pbar = tqdm(total=end_ts - start_ts, desc="Training Progress")
            self.clock.set_time(start_ts)
            current_ts = start_ts
            all_results = []

            self.accumulated_bids = []
            self.hours_accumulated = 0
            self.waiting_for_response = False
            self.waiting_for_next_day = False
            self.final_cycle_in_progress = False

            try:
                while current_ts < end_ts:
                    if self.eval_completed:
                        break

                    if self.waiting_for_next_day:
                        while self.waiting_for_next_day:
                            if hasattr(self, "event_mgr") and getattr(self.event_mgr, "next_day_event", None):
                                if self.event_mgr.next_day_event.is_set():
                                    self.waiting_for_next_day = False
                                    self.event_mgr.next_day_event.clear()
                                    break
                            if hasattr(self, "event_mgr") and self.event_mgr.stop_requested:
                                pbar.close()
                                return all_results
                            await asyncio.sleep(0.1)

                    step_result = await self.perform_single_step(current_ts)
                    all_results.append(step_result)
                    current_ts = step_result["next_timestamp"]
                    pbar.update(current_ts - step_result["timestamp"])

                    if current_ts >= end_ts and self.hours_accumulated > 0:
                        self.final_cycle_in_progress = True
                        try:
                            if self.hours_accumulated > 0:
                                self.event_mgr.current_bids = self.accumulated_bids.copy()
                                self.event_mgr.bid_ready_event.set()
                                if not self.waiting_for_response:
                                    self.waiting_for_response = True
                                    await self._wait_for_orderbook()
                                else:
                                    while self.waiting_for_response:
                                        await asyncio.sleep(0.1)
                                self.final_cycle_in_progress = False
                        except Exception as e:
                            logger.error("Error during final cycle: %s", e)
                            logger.error(traceback.format_exc())
                            self.final_cycle_in_progress = False
                            self.waiting_for_response = False
                            if hasattr(self, "event_mgr"):
                                self.event_mgr.orderbook_received_event.clear()
                                self.event_mgr.bid_ready_event.clear()
            finally:
                pbar.close()

                try:
                    if self._forecast_queue is not None:
                        await self._forecast_queue.put(None)
                    if self._forecast_worker_task is not None:
                        await self._forecast_worker_task
                except Exception:
                    pass
                finally:
                    self._forecast_queue = None
                    self._forecast_worker_task = None

            if self.waiting_for_response:
                while self.waiting_for_response:
                    await asyncio.sleep(0.1)

            if hasattr(self, "current_episode") and self.current_episode is not None:
                if hasattr(self, "learning_role") and self.learning_role.evaluation_mode and self.current_episode >= self.learning_role.training_episodes:
                    self.notify_training_complete()
                else:
                    self.notify_episode_complete(self.current_episode)

            return all_results

    # -------------------------- Orderbook waiting --------------------------
    async def _wait_for_orderbook(self):
        """
        Wait for and process returned orderbook from Java API.
        
        Blocks until event_mgr.orderbook_received_event is set,
        then calls process_orderbook(). Manages waiting_for_response flag.
        Logs detailed timing breakdown.
        
        Returns:
            None
        """


        try:
            if self.eval_completed:
                self.waiting_for_response = False
                return

            if not self.event_mgr.orderbook_received_event.is_set():
                while not self.event_mgr.orderbook_received_event.is_set():
                    await asyncio.sleep(0.1)

            if self.event_mgr.stop_requested:
                return

            start_process = time.time()
            if self.event_mgr.current_orderbook:
                await self.process_orderbook(self.event_mgr.current_orderbook)
                self.event_mgr.current_orderbook = None
                self.waiting_for_next_day = True
        except Exception as e:
            logger.error("Error in _wait_for_orderbook: %s", e)
        finally:
            start_finally = time.time()
            if not self.final_cycle_in_progress:
                self.accumulated_bids = []
                self.hours_accumulated = 0
            self.waiting_for_response = False
            if hasattr(self, "event_mgr"):
                self.event_mgr.orderbook_received_event.clear()

    # -------------------------- Bids --------------------------
    async def get_rl_bids(self) -> list:
        """
        Generate bids from RL agents for current clearing.
        
        Calculates bids for all units under "Operator-RL" using their
        RL strategies. Accumulates bids hourly; when 24 hours collected,
        fills missing hours with zero bids and triggers orderbook submission to market clearing.
        
        Returns:
            list: List of bid dicts with unit_id, price, volume, times
        """
        
        if self.eval_completed:
            return []
        bids = []
        current_time = timestamp2datetime(self.clock.time)
        market_config = next((cfg for name, cfg in self.markets.items() if name.startswith("EOM")), None)
        if market_config is None:
            return []

        product_start = current_time
        product_end = product_start + timedelta(hours=1)
        products = [(product_start, product_end)]

        if "Operator-RL" in self.unit_operators:
            for unit_id, unit in self.unit_operators["Operator-RL"].units.items():
                rl_strategy = None
                for market_id, strategy in unit.bidding_strategies.items():
                    if market_id in market_config.market_id and hasattr(strategy, "get_actions"):
                        rl_strategy = strategy
                        break
                if rl_strategy:
                    try:
                        orderbook = unit.calculate_bids(market_config=market_config, product_tuples=products)
                        total_volume, weighted_price = 0.0, 0.0
                        for order in orderbook:
                            # Aggregate into one hourly bid because the upstream API expects currently
                            # one bid per unit and hour.
                            vol = float(order["volume"])
                            total_volume += vol
                            weighted_price += float(order["price"]) * vol
                        if total_volume > 0:
                            bids.append(
                                {
                                    "unit_id": unit_id,
                                    "price": weighted_price / total_volume,
                                    "volume": total_volume,
                                    "start_time": str(current_time),
                                    "end_time": str(current_time + timedelta(hours=1)),
                                    "strategy_type": "pp_learning",
                                }
                            )
                    except Exception as e:
                        logger.error("Error calculating bids for unit %s: %s", unit_id, e)

        if hasattr(self, "event_mgr"):
            self.accumulated_bids.extend(bids)
            self.hours_accumulated += 1
            if self.hours_accumulated >= 24:
                bids_by_unit = {}
                for bid in self.accumulated_bids:
                    uid = bid.get("unit_id")
                    bids_by_unit.setdefault(uid, {})
                    start_time = bid.get("start_time")
                    if start_time:
                        hour = int(start_time.split()[1][:2])
                        bids_by_unit[uid][hour] = bid

                filled_bids = []
                date_part = current_time.strftime("%Y-%m-%d")
                for uid, hour_bids in bids_by_unit.items():
                    for h in range(24):
                        if h not in hour_bids:
                            filled_bids.append(
                                {
                                    "unit_id": uid,
                                    "price": 0.0,
                                    "volume": 0.0,
                                    "start_time": f"{date_part} {h:02d}:00:00",
                                    "end_time": f"{date_part} {(h + 1) % 24:02d}:00:00",
                                    "strategy_type": "pp_learning",
                                }
                            )
                        else:
                            filled_bids.append(hour_bids[h])

                self.accumulated_bids = filled_bids
                self.event_mgr.current_bids = self.accumulated_bids.copy()
                self.event_mgr.bid_ready_event.set()
                self.waiting_for_response = True
                asyncio.create_task(self._wait_for_orderbook())

        return bids

    # -------------------------- Orderbook processing --------------------------
    async def process_orderbook(self, orderbook: dict) -> None:
        """
        Process market results and compute rewards.
        
        Parses orderbook with market prices and accepted bids, calculates
        profit/regret/costs, updates RL policy, logs to CSV. Handles
        final evaluation cycle detection.
        
        Args:
            orderbook (dict): Orderbook with keys:
                - market_prices: List of price points
                - accepted_bids: List of executed bids
        
        Returns:
            None
        """
        logger.info("--- PROCESS ORDERBOOK ---")
        daily_profit = 0.0
        daily_regret = 0.0
        daily_costs = 0.0
        start_time = None

        try:
            t_market_config = time.time()
            market_config = next((cfg for name, cfg in self.markets.items() if name.startswith("EOM")), None)
            if market_config is None:
                return

            t_orderbook = time.time()
            assume_orderbook = []
            bid_hours = set()
            date_ref = None

            for price_entry in orderbook.get("market_prices", []):
                start_time = datetime.fromisoformat(price_entry.get("start_time"))
                if date_ref is None:
                    date_ref = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                bid_hours.add(start_time.hour)

            for bid in orderbook.get("accepted_bids", []):
                st = datetime.fromisoformat(bid.get("start_time"))
                if date_ref is None:
                    date_ref = st.replace(hour=0, minute=0, second=0, microsecond=0)
                bid_hours.add(st.hour)
                assume_orderbook.append(
                    {
                        "start_time": st,
                        "end_time": datetime.fromisoformat(bid.get("end_time")),
                        "price": bid.get("price", 0),
                        "volume": bid.get("volume", 0),
                        "accepted_volume": bid.get("accepted_volume", 0),
                        "accepted_price": bid.get("accepted_price", 0),
                        "unit_id": bid.get("unit_id", ""),
                        "node": bid.get("node", "default_node"),
                        "hour": st.hour,
                    }
                )

            t_complete = time.time()
            if not assume_orderbook:
                return

            all_units = set(b["unit_id"] for b in assume_orderbook)
            complete_orderbook = assume_orderbook[:]
            for uid in all_units:
                ref_bid = next((b for b in assume_orderbook if b["unit_id"] == uid), None)
                if not ref_bid:
                    continue
                for h in range(24):
                    if h not in bid_hours:
                        zt = date_ref + timedelta(hours=h)
                        complete_orderbook.append(
                            {
                                "start_time": zt,
                                "end_time": zt + timedelta(hours=1),
                                "price": 0,
                                "volume": 0,
                                "accepted_volume": 0,
                                "accepted_price": 0,
                                "unit_id": uid,
                                "node": ref_bid["node"],
                                "hour": h,
                            }
                        )

            complete_orderbook.sort(key=lambda x: x["hour"])
            for bid in complete_orderbook:
                bid.pop("hour", None)

            t_rewards = time.time()
            if "Operator-RL" in self.unit_operators:
                for unit_id, unit in self.unit_operators["Operator-RL"].units.items():
                    unit_bids = [b for b in complete_orderbook if b["unit_id"] == unit_id]
                    if not unit_bids:
                        continue

                    time_grouped = {}
                    for b in unit_bids:
                        st = b["start_time"]
                        time_grouped.setdefault(st, []).append(b)

                    for st, hourly_bids in time_grouped.items():
                        for market_id, strategy in unit.bidding_strategies.items():
                            if market_id in market_config.market_id and hasattr(strategy, "calculate_reward"):
                                strategy.calculate_reward(unit=unit, marketconfig=market_config, orderbook=hourly_bids)
                                hour_profit = unit.outputs["profit"].loc[st]
                                hour_regret = unit.outputs["regret"].loc[st]
                                hour_costs = unit.outputs["total_costs"].loc[st]
                                daily_profit += hour_profit
                                daily_regret += hour_regret
                                daily_costs += hour_costs

                daily_reward = 0.0
                for unit_id, unit in self.unit_operators["Operator-RL"].units.items():
                    if "rl_rewards" in unit.outputs and unit.outputs["rl_rewards"]:
                        daily_reward += sum(unit.outputs["rl_rewards"])

                self.current_episode_reward += daily_reward

                if hasattr(self, "learning_role") and not self.learning_role.evaluation_mode:
                    try:
                        await self.update_rl_policy(complete_orderbook)
                    except Exception as e:
                        logger.error("Policy update failed: %s", e)

        finally:
            t_finally = time.time()
            try:
                self.training_state["days_processed"] += 1
                self.training_state["total_rewards"] += daily_profit
                self.training_state["best_reward"] = max(self.training_state["best_reward"], daily_profit)

                csv_path = self.export_csv_path + f"\episode.csv"
                import os, csv

                day_index = self.training_state["days_processed"]
                current_episode = getattr(self, "current_episode", 0)
                date = start_time.strftime("%Y-%m-%d") if start_time else ""
                file_exists = os.path.isfile(csv_path)

                episode_type = "Evaluation" if hasattr(self,
                                                       "learning_role") and self.learning_role.evaluation_mode else "Training"

                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if not file_exists:
                        w.writerow(["Episode", "Episode_Type", "Training Day", "Date", "Total_Reward", "Daily_Profit",
                                    "Daily_Regret", "Daily_Costs"])
                    w.writerow([current_episode, episode_type, day_index, date, self.training_state["total_rewards"],
                                daily_profit, daily_regret, daily_costs])

                current_ts = self.clock.time
                final_day_in_eval_mode = (
                        hasattr(self, "learning_role")
                        and getattr(self, "evaluation_mode")
                        and hasattr(self, "end_ts")
                        and current_ts >= self.end_ts
                )
                if final_day_in_eval_mode and self.learning_config["training_episode"] == current_episode:
                    self.waiting_for_next_day = False
                    self.notify_training_complete()
                    self.final_cycle_in_progress = True
                    self.eval_completed = True
                logger.info("Orderbook processed")
            except Exception as e:
                logger.error("Error updating training state: %s", e)


    # -------------------------- Notify helpers --------------------------
    def notify_episode_complete(self, episode_num: int):
        try:
            url = f"{self.java_api_url}/session/{self.session_id}/episode_complete"
            r = httpx.post(url, json={"episode": int(episode_num)})
            if r.status_code == 200:
                logger.info("Episode %s completed", episode_num)
            else:
                logger.error("Episode completion notify failed: %s", r.status_code)
        except Exception as e:
            logger.error("Episode completion notify error: %s", e)
            logger.error(traceback.format_exc())

    def notify_training_complete(self):
        try:
            url = f"{self.java_api_url}/session/{self.session_id}/training_complete"
            r = httpx.post(url, json={"status": "complete"})
            if r.status_code == 200:
                logger.info("Training completion notification sent")
            else:
                logger.error("Training completion notify failed: %s", r.status_code)
        except Exception as e:
            logger.error("Training completion notify error: %s", e)
            logger.error(traceback.format_exc())

    # -------------------------- Output agent --------------------------
    def setup_output_agent(self, save_frequency_hours: int, episode: int, eval_episode: int) -> None:
        """
        Configure and register output writer agent.
        
        Creates WriteOutput role for database/CSV export, initializes
        RL parameter table in database, and registers agent in container.
        
        Args:
            save_frequency_hours (int): Output save interval (hours)
            episode (int): Current training episode
            eval_episode (int): Current evaluation episode
        
        Returns:
            None
        """
        
        logger.debug(
            "PowerSUME: creating output agent db=%s export_csv_path=%s test_mode=%s",
            self.db_uri,
            self.export_csv_path,
            self.test_mode,
        )

        self.output_role = WriteOutput(
            simulation_id=self.simulation_id,
            start=self.start,
            end=self.end,
            db_uri=self.db_uri,
            export_csv_path=self.export_csv_path,
            save_frequency_hours=save_frequency_hours,
            learning_mode=self.learning_mode,
            evaluation_mode=self.evaluation_mode,
            episode=episode,
            eval_episode=eval_episode,
            additional_kpis=self.additional_kpis,
            outputs_buffer_size_mb=self.scenario_data["config"].get("outputs_buffer_size_mb", 300),
        )
        if not self.output_agent_addr:
            return

        if self.db_uri:
            engine = create_engine(self.db_uri)
            with engine.begin() as conn:
                conn.execute(
                    text("CREATE TABLE IF NOT EXISTS rl_params (id INTEGER PRIMARY KEY, simulation TEXT,unit TEXT,reward REAL,timestamp INTEGER)")
                )

                if not self.output_agent_addr:
                    return

        if not self.db_uri and not self.export_csv_path:
            self.output_agent_addr = None
        else:
            self.output_agent_addr = addr(self.addr, "export_agent_1")

        import sys

        if sys.platform == "linux" and self.distributed_role is not None:
            super().setup_output_agent(simulation_id, save_frequency_hours)  # type: ignore[name-defined]
        else:
            output_agent = agent_composed_of(
                self.output_role,
                register_in=self.container,
                suggested_aid=self.output_agent_addr.aid,
            )
            output_agent.suspendable_tasks = False

    # -------------------------- RL policy update & helpers --------------------------
    async def update_rl_policy(self, orderbook: list):
        """
        Update RL agent policies using collected experience.
        
        Gathers observations, actions, rewards from all RL units,
        reshapes for replay buffer, adds to buffer, calls policy update
        unless in initial experience collection phase.
        
        Args:
            orderbook (list): Market results for hourly learning output
        
        Returns:
            None
        """
        
        episodes_collecting_initial = self.learning_role.episodes_collecting_initial_experience

        rl_operator = self.unit_operators.get("Operator-RL")
        if not rl_operator:
            return

        total_units = len(rl_operator.units)
        all_obs, all_actions, all_rewards = [], [], []

        async def process_unit(unit):
            if not hasattr(unit, "outputs"):
                return [], [], []
            observations = unit.outputs.get("rl_observations", [])
            actions = unit.outputs.get("rl_actions", [])
            rewards = unit.outputs.get("rl_rewards", [])
            obs_list, act_list, rew_list = [], [], []
            if len(observations) > 0 and len(actions) > 0 and len(rewards) > 0:
                steps = min(len(observations), len(actions), len(rewards))
                for i in range(steps):
                    obs_list.append(observations[i])
                    act_list.append(actions[i])
                    rew_list.append(rewards[i])
            # Reset outputs after transfer
            unit.outputs["rl_observations"] = []
            unit.outputs["rl_actions"] = []
            unit.outputs["rl_rewards"] = []
            return obs_list, act_list, rew_list

        results = await asyncio.gather(
            *[process_unit(unit) for unit in rl_operator.units.values()]
        )
        for obs_list, act_list, rew_list in results:
            all_obs.extend(obs_list)
            all_actions.extend(act_list)
            all_rewards.extend(rew_list)

        if all_obs and all_actions and all_rewards:
            obs_dim = len(all_obs[0])
            act_dim = len(all_actions[0])
            timesteps = len(all_obs) // total_units

            # Arrays for ReplayBuffer (NumPy, CPU)
            reshaped_obs = np.zeros((timesteps, total_units, obs_dim), dtype=np.float32)
            reshaped_actions = np.zeros((timesteps, total_units, act_dim), dtype=np.float32)
            reshaped_rewards = np.zeros((timesteps, total_units), dtype=np.float32)

            def to_numpy_cpu(x):
                if isinstance(x, (list, tuple)):
                    x = np.array(x, dtype=np.float32)
                elif hasattr(x, "detach"):  # torch.Tensor
                    if x.dim() == 0:
                        return float(x.detach().cpu().item())
                    return x.detach().cpu().numpy()
                return np.array(x, dtype=np.float32)

            for t in range(timesteps):
                for u in range(total_units):
                    idx = u * timesteps + t
                    if idx < len(all_obs):
                        reshaped_obs[t, u] = to_numpy_cpu(all_obs[idx])
                        reshaped_actions[t, u] = to_numpy_cpu(all_actions[idx])
                        reshaped_rewards[t, u] = float(to_numpy_cpu(all_rewards[idx]))

            self.learning_role.buffer.add(
                obs=reshaped_obs, actions=reshaped_actions, reward=reshaped_rewards
            )

            try:
                hourly_groups: dict[pd.Timestamp, list] = {}
                for order in orderbook:
                    st = order.get("start_time")
                    if isinstance(st, str):
                        st = pd.Timestamp(st)
                    hourly_groups.setdefault(st, []).append(order)

                for st, hourly_orders in sorted(hourly_groups.items()):
                    rl_operator.write_learning_to_output(hourly_orders, "EOM")

            except Exception as e:
                logger.error("Hourly write_learning_to_output failed: %s", e)


            self.learning_role.update_policy()
            if hasattr(self, "track_nn_weight_stats"):
                _ = self.track_nn_weight_stats()

    # -------------------------- Reset & buffer debug --------------------------
    def track_nn_weight_stats(self):
        """
        Track neural network weight changes during training.
        
        Monitors actor network parameters across episodes, detects
        significant weight updates, and logs statistics for debugging.
        
        Returns:
            dict: Statistics with keys:
                - has_changes: Whether significant updates detected
                - update_number: Training update counter
                - stats: Per-unit parameter change details
        """
        
        try:
            if not hasattr(self, "previous_weights"):
                self.previous_weights = {}
                self.weight_change_counter = 0

            self.weight_change_counter += 1
            stats = {}
            change_detected = False

            for unit_id, strategy in self.learning_role.rl_strats.items():  # type: ignore[attr-defined]
                if not hasattr(strategy, "actor"):
                    continue
                current_weights = {}
                changes = []
                for i, param in enumerate(strategy.actor.parameters()):
                    tensor = param.detach().cpu()
                    param_id = f"layer_{i}"
                    mean_val = float(tensor.mean())
                    norm_val = float(tensor.norm())
                    current_weights[param_id] = {"mean": mean_val, "norm": norm_val}
                    if unit_id in self.previous_weights and param_id in self.previous_weights[unit_id]:
                        prev = self.previous_weights[unit_id][param_id]
                        mean_change = abs(mean_val - prev["mean"])
                        norm_change = abs(norm_val - prev["norm"])
                        significant = mean_change > 1e-4
                        change_detected |= significant
                        changes.append({"mean_change": mean_change, "norm_change": norm_change, "significant": significant})
                self.previous_weights[unit_id] = current_weights
                stats[unit_id] = {"changes": changes}

            return {"has_changes": change_detected, "update_number": self.weight_change_counter, "stats": stats}
        except Exception as e:
            logger.error("track_nn_weight_stats error: %s", e)
            return {"error": str(e)}

    def reset(self):
        """
        Reset simulation state for new episode.
        
        Clears bid accumulation, resets waiting flags, resets training
        state counters, clears event flags, calls parent reset().
        
        Returns:
            bool: Always True
        """
        
        self.accumulated_bids = []
        self.hours_accumulated = 0
        self.waiting_for_response = False
        self.training_state["days_processed"] = 0
        self.current_episode_reward = 0.0
        if hasattr(self, "final_cycle_in_progress"):
            self.final_cycle_in_progress = False
        if hasattr(self, "event_mgr"):
            self.event_mgr.bid_ready_event.clear()
            self.event_mgr.orderbook_received_event.clear()
        super().reset()
        return True


