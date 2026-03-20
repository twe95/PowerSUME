from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import logging
import os
import threading
import time
import uuid

from assume.common.fast_pandas import FastSeries
from pydantic import Field

from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, Request
from pydantic import BaseModel
import orjson
import uvicorn


#from ApiForecaster import ApiForecaster, UnitScopedForecaster
from power_learning_01 import power_run_learning
from power_learning_01 import power_run_evaluation
from powerworld import PowerWorld
from assume.scenario.loader_csv import load_scenario_folder



# Configure logging 251.26927.90
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PowerSUME API", description="API for PowerSUME reinforcement learning")

# Global session tracking
active_sessions = {}

class SessionInitRequest(BaseModel):
    
    """Configuration model for initializing a new PowerSUME learning session.
    
    This model defines all parameters needed to set up a new reinforcement learning
    session that interfaces with simulation market clearing.
    
    Attributes:
        inputs_path (str): Path to scenario input files
        scenario (str): Name of the scenario to run
        study_case (str): Specific study case within the scenario
        java_api_url (str): URL of the market simulation API endpoint
        test_mode (bool): Whether to run in test mode without external communication
        verbose (bool): Whether to enable verbose logging
        continue_learning (bool): Whether to continue training from saved policies
        evaluation_mode (bool): Whether to run in evaluation-only mode
        initial_experience_episodes (int): Number of episodes for initial random exploration
        validation_interval (int): Number of episodes between validation runs
    """
    output_directory: str
    inputs_path: str
    scenario: str
    study_case: str
    java_api_url: str = "http://localhost:8000"
    test_mode: bool = False
    verbose: bool = False
    evaluation_mode: bool = False
    forecastUpdateFrequency: int = 14



class OrderbookSubmission(BaseModel):

    """Data model for market orderbook results from simulation.
    
    Contains the results of market clearing, including accepted bids and market prices,
    which are used to calculate rewards for the RL agent.
    
    Attributes:
        accepted_bids (List[Dict[str, Any]]): List of bids that were accepted in the market
        market_prices (List[Dict[str, Any]]): List of cleared market prices for each hour
    """

    accepted_bids: List[Dict[str, Any]]
    market_prices: List[Dict[str, Any]]


### FORECAST ──────────────────────────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    unit_id: str
    start_time: str  # ISO8601 from market simulation
    end_time: str    # ISO8601 from market simulation
    residual_load_EOM: Optional[float] = None
    availability: Optional[float] = None
    price_EOM: Optional[float] = None
    varCosts: Optional[float] = None

class ForecastData(BaseModel):
    """Complete forecast batch from market simulation."""
    forecast_data: List[ForecastPoint] = Field(default_factory=list)
### FORECAST END ──────────────────────────────────────────────────────────────────

class ForecastArrayPayload(BaseModel):
    unit_ids: List[int]
    start_epoch: int    # Startzeitpunkt (Sekunden seit 1970-01-01 UTC)
    step_sec: int       # Schrittweite (meist 3600)
    len: int            # Anzahl Zeitschritte
    rl: List[float]     # Residual load
    av: List[float]     # Availability
    vc: List[float]     # Variable costs
    p: List[float]      # Price_EOM

class SessionEventManager:

    """Event manager for synchronizing communication between market simulation and the RL agent.
    
    Manages threading events that coordinate the flow of data and execution between
    the FastAPI server, PowerWorld simulation, and market simulation market clearing.
    
    Attributes:
        ready_for_params_event (threading.Event): Signals when the agent is ready to receive parameters
        bid_ready_event (threading.Event): Signals when new bids are ready for market simulation
        orderbook_received_event (threading.Event): Signals when market results are received
        next_day_event (threading.Event): Signals to proceed to the next simulation day
        next_episode_event (threading.Event): Signals to start the next training episode
        current_bids (list): Storage for current bid data to send to market simulation
        current_orderbook (dict): Storage for latest orderbook results
        stop_requested (bool): Flag to terminate the training process
        episode_waiting (bool): Flag indicating waiting for next episode signal
        evaluation_episode (bool): Flag indicating current episode is for evaluation
    """

    def __init__(self):
        self.bid_ready_event = threading.Event()
        self.orderbook_received_event = threading.Event()
        self.next_day_event = threading.Event()
        self.next_episode_event = threading.Event()
        self.current_bids = []
        self.current_orderbook = None
        self.stop_requested = False
        self.episode_waiting = False
        self.evaluation_episode = False
        self.current_params = {}

        ### FORECAST ──────────────────────────────────────────────────────────────
        # Thread-sicherer Buffer: unit_id -> ts(str) -> {metric: value}
        self._forecast_lock = threading.Lock()
        self._forecast_store: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
        self.forecast_updated_event = threading.Event() #data received
        self.forecast_applied_event = threading.Event() #data processed
        # Cache für den zuletzt empfangenen Forecast pro Unit (Rohwerte)
        # Struktur: { unit_id: {"availability": [..], "price_EOM": [..], "residual_load_EOM": [..], "fuel_price": [..], "fuel_key": str } }
        self._last_forecast_cache: Dict[str, Dict[str, Any]] = {}
        ### FORECAST END ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running.
    
    Returns:
        dict: Status information with current timestamp
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}



@app.post("/session/init")
async def initialize_session(request: SessionInitRequest, background_tasks: BackgroundTasks):

    """Initialize a new PowerSUME learning session.
    
    Creates a new session with unique ID, sets up the PowerWorld environment,
    loads scenario data, and starts the learning process in a background thread.
    
    Args:
        request (SessionInitRequest): Configuration parameters for the session
        background_tasks (BackgroundTasks): FastAPI background tasks handler
        
    Returns:
        dict: Session information with ID and initialization status
        
    Raises:
        HTTPException: If initialization fails
    """

    session_id = str(uuid.uuid4())
    
    # Initialize event manager for this session
    event_mgr = SessionEventManager()
    logger.info(f"API EVENT_MGR ID: {id(event_mgr)}")

    active_sessions[session_id] = {
        "config": request.dict(),
        "status": "initializing",
        "event_mgr": event_mgr,
        "start_time": datetime.now().isoformat()
    }

    try:
        directories = request.output_directory
        # Set up directories
        csv_path = directories+ f"outputs/{session_id}"
        os.makedirs(f"local_db", exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)
        
        # Configure database
        db_uri = f"sqlite:///{directories}/local_db/assume_db_{session_id}.db"
        
        # Initialize PowerWorld
        powerworld = PowerWorld(
            java_api_url=request.java_api_url,
            addr="world",
            test_mode=request.test_mode,
            export_csv_path=csv_path,
            database_uri=db_uri,
            session_id=session_id,
            forecast_update_frequency = request.forecastUpdateFrequency,
        )
        
        # Store event manager in PowerWorld for synchronization
        powerworld.event_mgr = event_mgr  # Ensure this is the SAME object

        # Add evaluation_mode flag to PowerWorld object
        powerworld.evaluation_mode = request.evaluation_mode
        logger.info(f"Setting evaluation_mode={request.evaluation_mode} in PowerWorld")

        # Load scenario data - this will load learning configuration from files
        load_scenario_folder(
            powerworld,
            inputs_path=request.inputs_path,
            scenario=request.scenario,
            study_case=request.study_case
        )

        # Store PowerWorld in session
        session = active_sessions[session_id]
        session["powerworld"] = powerworld
        session["status"] = "running"
        
        # Start learning process in background thread
        background_tasks.add_task(
            run_powersume_learning_session,
            session_id,
            request,
            powerworld,
            event_mgr
        )
        
        return {
            "session_id": session_id,
            "status": "initializing",
            "message": "Session initializing, check status endpoint for updates"
        }
        
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        active_sessions[session_id]["status"] = "failed"
        active_sessions[session_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):

    """Get the current status of a PowerSUME learning session.
    
    Returns information about the session including current episode progress.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Session status information
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    powerworld = session.get("powerworld")
    
    # Get current episode from PowerWorld if available
    current_episode = getattr(powerworld, "current_episode", 0) if powerworld else 0
    total_episodes = 0
    if powerworld and hasattr(powerworld, "learning_role"):
        total_episodes = getattr(powerworld.learning_role, "training_episodes", 100)
    
    return {
        "session_id": session_id,
        "status": session["status"],
        "start_time": session["start_time"],
        "current_episode": current_episode,
        "total_episodes": total_episodes
    }

@app.post("/session/{session_id}/begin_next_day")
async def begin_next_day(session_id: str):

    """Signal to start the next day's simulation in PowerWorld.
    
    Called by market simulation to advance the learning to the next day.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """


    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]

    # Signal PowerWorld to proceed to next day
    event_mgr.next_day_event.set()

    return {"status": "accepted", "message": "Next day signal sent"}


@app.post("/session/{session_id}/episode_complete")
async def episode_complete(session_id: str, data: dict = Body(...)):

    """Handle notification that an episode has completed.
    
    Updates session status and waits for signal to begin next episode.
    
    Args:
        session_id (str): Unique identifier for the session
        data (dict): Information about the completed episode
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    episode = data.get("episode", 0)

    #Update session status
    session["status"] = "episode_complete"
    session["current_episode"] = episode
    session["waiting_next_episode"] = True
    event_mgr.episode_waiting = True

    logger.info(f" PowerSUME Episode {episode} completed for session {session_id}")

    return {"status": "accepted", "message": "Episode completion acknowledged"}


@app.post("/session/{session_id}/begin_next_episode")
async def begin_next_episode(session_id: str):

    """Signal to start the next training episode.
    
    Called by market simulation to initiate the next episode after episode_complete.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]

    # Set signal to start the next episode
    event_mgr.next_episode_event.set()
    event_mgr.episode_waiting = False
    session["status"] = "running"
    session["waiting_next_episode"] = False

    return {"status": "accepted", "message": "Next episode signal sent"}


@app.post("/session/{session_id}/training_complete")
async def training_complete(session_id: str):

    """Handle notification that training and evaluation is complete.
    
    Sets flags to terminate all processes and marks session as completed.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Received training completion notification for session {session_id}")

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]

    # Set stop flag to terminate any background processes
    event_mgr.stop_requested = True

    # Set all event flags to unblock any waiting processes
    event_mgr.orderbook_received_event.set()
    event_mgr.bid_ready_event.set()
    event_mgr.next_day_event.set()
    event_mgr.next_episode_event.set()
    event_mgr.forecast_updated_event.set()

    # Update session status
    session["status"] = "completed"

    logger.info(f"Training completion acknowledged for session {session_id}")

    return {"status": "accepted", "message": "Training completion acknowledged"}


@app.get("/session/{session_id}/bids")
async def get_bids(session_id: str):

    """Retrieve the latest bids generated by the RL agent.
    
    Called by market simulation to get bids when they're ready.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Status and bid data if available
        
    Raises:
        HTTPException: If session is not found
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    
    # Check if bids are ready
    if not event_mgr.bid_ready_event.is_set():
        return {
            "status": "waiting",
            "message": "No bids ready yet",
            "bids": [],
            "timestamp": datetime.now().isoformat(),
        }
    
    # Return bids and reset the event for next day
    bids = event_mgr.current_bids.copy()
    event_mgr.bid_ready_event.clear()  # Reset for next day

    return {
        "status": "ready",
        "bids": bids,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/session/{session_id}/orderbook")
async def submit_orderbook(session_id: str, orderbook: OrderbookSubmission):

    """Submit market clearing results to the RL agent.
    
    Called by market simulation to provide market results after clearing.
    
    Args:
        session_id (str): Unique identifier for the session
        orderbook (OrderbookSubmission): Market clearing results
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        logger.error(f"Session {session_id} not found in active_sessions!")
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    orderbook_dict = orderbook.dict()

    # Store the orderbook
    event_mgr.current_orderbook = orderbook_dict

    event_mgr.orderbook_received_event.set()

    return {
        "status": "accepted",
        "message": "Orderbook received and processing"
    }


### FORECAST ENDPOINT ─────────────────────────────────────────────────────────────
@app.post("/session/{session_id}/forecast")
async def submit_forecast_fast(session_id: str, request: Request):
    """
    Fast forecast ingestion endpoint (Variant 2: Queue + Worker + Prestart Buffer).
    Erweitert: Wartet bis Units existieren (max 20s), damit frühe Forecasts nicht verloren gehen.
    """
    logger.info("Received forecast submission for session %s", session_id)
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        raw = await request.body()
        payload = orjson.loads(raw)
    except Exception as e:
        logger.exception("invalid JSON")
        return {"status": "error", "message": f"invalid JSON: {e}"}

    units_payload = payload.get("units")
    if not isinstance(units_payload, list) or len(units_payload) == 0:
        return {"status": "error", "message": "payload must contain non-empty 'units' array"}

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    pw: PowerWorld = session.get("powerworld")
    if pw is None:
        return {"status": "error", "message": "PowerWorld not initialized"}

    try:
        unit_wait_timeout = float(request.query_params.get("unit_wait", 20.0))
    except Exception:
        unit_wait_timeout = 20.0
    unit_poll_interval = 0.25

    async def await_unit(operator_id: str, uid_raw, timeout: float) -> tuple[object | None, str]:
        """Warte bis eine Unit (id oder str(id)) unter operator_id existiert oder Timeout erreicht ist."""
        start = time.time()
        uid_str = str(uid_raw)
        while (time.time() - start) < timeout:
            op = pw.unit_operators.get(operator_id)
            if op and hasattr(op, "units"):
                unit = op.units.get(uid_raw) or op.units.get(uid_str)
                if unit is not None:
                    return unit, operator_id
            alt_id = "Operator_RL" if operator_id == "Operator-RL" else "Operator-RL"
            op_alt = pw.unit_operators.get(alt_id)
            if op_alt and hasattr(op_alt, "units"):
                unit = op_alt.units.get(uid_raw) or op_alt.units.get(uid_str)
                if unit is not None:
                    return unit, alt_id
            await asyncio.sleep(unit_poll_interval)
        return None, operator_id

    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    pw: PowerWorld = session.get("powerworld")
    if pw is None:
        return {"status": "error", "message": "PowerWorld not initialized"}

    event_mgr.forecast_applied_event.clear()

    def _conform(values: list[float], idx_len: int, name: str, uid: int) -> list[float]:
        vlen = len(values)
        if vlen > idx_len:
            logger.info("Forecast %s für Unit %s ist länger (%s>%s). Kappe auf Indexlänge.",
                        name, uid, vlen, idx_len)
            return values[:idx_len]
        if vlen < idx_len:
            logger.info("Forecast %s für Unit %s ist kürzer (%s<%s). Fülle mit letztem Wert auf.",
                        name, uid, vlen, idx_len)
            if vlen == 0:
                fill = 0.0
                return [fill] * idx_len
            fill = values[-1]
            return values + [fill] * (idx_len - vlen)
        return values

    try:
        with event_mgr._forecast_lock:
            for u in units_payload:
                uids = u["unit_ids"]
                if isinstance(uids, int):
                    uids = [uids]
                if not (isinstance(uids, list) and all(isinstance(x, int) for x in uids) and len(uids) > 0):
                    return {"status": "error", "message": "unit_ids must be a non-empty list of integers"}

                try:
                    rl_vals = list(map(float, u["rl"]))
                    av_vals = list(map(float, u["av"]))
                    vc_vals = list(map(float, u["vc"]))
                    p_vals  = list(map(float, u["p"]))
                except Exception:
                    return {"status": "error", "message": "rl/av/vc/p must be numeric arrays"}

                for uid in uids:
                    unit, used_operator = await await_unit("Operator-RL", uid, unit_wait_timeout)
                    if unit is None:
                        logger.warning("Unit %s not found under Operator-RL nach %.1fs – skipping", uid, unit_wait_timeout)
                        continue

                    f = getattr(unit, "forecaster", None)
                    if f is None:
                        logger.warning("Unit %s has no forecaster; skipping", uid)
                        continue

                    idx = getattr(f, "index", None)
                    if idx is None:
                        logger.info("Unit %s forecaster has no index; skipping", uid)
                        continue

                    try:
                        idx_len = len(idx)
                    except Exception:
                        logger.info("Unit %s forecaster index length unknown; skipping", uid)
                        continue

                    av_ready = _conform(av_vals, idx_len, "availability", uid)
                    p_ready  = _conform(p_vals,  idx_len, "price_EOM", uid)
                    rl_ready = _conform(rl_vals, idx_len, "residual_load_EOM", uid)
                    vc_ready = _conform(vc_vals, idx_len, "fuel_price", uid)

                    try:
                        av_series = FastSeries(index=idx, value=av_ready, name="availability")
                        price_series = FastSeries(index=idx, value=p_ready,  name="price_EOM")
                        rl_series    = FastSeries(index=idx, value=rl_ready, name="residual_load_EOM")
                    except Exception as e:
                        return {"status": "error", "message": f"failed to build FastSeries (unit {uid}): {e}"}

                    fuel_key = None
                    try:
                        existing_fp = getattr(f, "fuel_prices", None)
                        if isinstance(existing_fp, dict) and len(existing_fp) > 0:
                            fuel_key = next(iter(existing_fp.keys()))
                    except Exception:
                        pass
                    if not fuel_key:
                        fuel_key = getattr(unit, "fuel", None) or getattr(unit, "fuel_type", None) or "fuel"

                    try:
                        fuel_series = FastSeries(index=idx, value=vc_ready, name=f"fuel_price_{fuel_key}")
                    except Exception as e:
                        return {"status": "error", "message": f"failed to build FastSeries for fuel_prices (unit {uid}): {e}"}

                    f.availability = av_series
                    f.price = {"EOM": price_series}
                    f.residual_load = {"EOM": rl_series}
                    f.fuel_prices = {str(fuel_key): fuel_series}

                    try:
                        event_mgr._last_forecast_cache[str(uid)] = {
                            "availability": av_ready,
                            "price_EOM": p_ready,
                            "residual_load_EOM": rl_ready,
                            "fuel_price": vc_ready,
                            "fuel_key": str(fuel_key),
                        }
                    except Exception:
                        pass

        event_mgr.forecast_applied_event.set()
        logger.info("Forecast data applied for session %s", session_id)
        return {"status": "accepted", "received_units": len(units_payload), "mode": "direct_write"}

    except KeyError as ke:
        return {"status": "error", "message": f"missing key: {ke}"}
    except Exception as e:
        logger.exception("failed to apply forecast directly")
        return {"status": "error", "message": f"failed to apply forecast: {e}"}



@app.post("/session/{session_id}/stop")
async def stop_session(session_id: str):

    """Request to stop an active session.
    
    Sets flags to terminate the session gracefully.
    
    Args:
        session_id (str): Unique identifier for the session
        
    Returns:
        dict: Status acknowledgment
        
    Raises:
        HTTPException: If session is not found
    """

    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    event_mgr = session["event_mgr"]
    
    event_mgr.stop_requested = True
    event_mgr.orderbook_received_event.set()
    
    return {"status": "stopping", "message": "Stop request sent to session"}

async def run_powersume_learning_session(
    session_id: str,
    request: SessionInitRequest,
    powerworld: PowerWorld,
    event_mgr: SessionEventManager
):

    """Run the PowerSUME learning process in a background thread.
    
    This function starts the actual learning process based on the session configuration,
    running either training or evaluation mode as requested.
    
    Args:
        session_id (str): Unique identifier for the session
        request (SessionInitRequest): Session configuration parameters
        powerworld (PowerWorld): The PowerWorld instance for this session
        event_mgr (SessionEventManager): Event manager for synchronization
    """


    try:
        def run_learning():
            if request.evaluation_mode:
                power_run_evaluation(
                    powerworld=powerworld

                )
            else:
                # Run in training mode
                power_run_learning(
                    powerworld=powerworld,
                    inputs_path=request.inputs_path
                )
            
        learning_thread = threading.Thread(target=run_learning)
        learning_thread.start()
        
        while learning_thread.is_alive():
            await asyncio.sleep(1)
            
            if event_mgr.stop_requested:
                active_sessions[session_id]["status"] = "completed"
                break
                
    except Exception as e:
        logger.error(f"Error in learning process: {e}")
        active_sessions[session_id]["status"] = "failed"
        active_sessions[session_id]["error"] = str(e)


def main():

    """Run the FastAPI app using uvicorn server.
    
    This is the entry point for starting the API server when running as a standalone script.
    """

    uvicorn.run("PowerSUME_api_01:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")

if __name__ == "__main__":
    main()