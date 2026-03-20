import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import time

from assume.common.exceptions import AssumeException
from assume.common.fast_pandas import FastSeries
from assume.reinforcement_learning.buffer import ReplayBuffer
from assume.scenario.loader_csv import setup_world

from powerworld import PowerWorld

logger = logging.getLogger(__name__)

def _repair_forecast_if_needed(powerworld: PowerWorld) -> None:
    """
    Repair forecast data if it was lost or overwritten after setup.
    
    Checks if each RL unit has valid forecast values (price['EOM'] != all 1000).
    If missing or invalid, re-applies cached forecast values from the API and
    calls prepare_observations on the unit's strategy.
    
    Args:
        powerworld (PowerWorld): The PowerWorld instance containing units and event manager
    
    Returns:
        None
    """
    evt_mgr = getattr(powerworld, "event_mgr", None)
    if evt_mgr is None:
        return
    cache = getattr(evt_mgr, "_last_forecast_cache", None)
    if not isinstance(cache, dict) or not cache:
        return

    op = powerworld.unit_operators.get("Operator-RL") or powerworld.unit_operators.get("Operator_RL")
    if not op:
        return

    for uid, unit in op.units.items():
        f = getattr(unit, "forecaster", None)
        if f is None:
            continue
        try:
            price_eom = getattr(f, "price", {}).get("EOM")
            needs_repair = False
            if price_eom is None:
                needs_repair = True
            else:
                values = None
                if hasattr(price_eom, "values"):
                    values = list(price_eom.values)
                elif hasattr(price_eom, "to_list"):
                    values = list(price_eom.to_list())
                if values is None:
                    needs_repair = True
                else:
                    if len(values) > 0 and all(abs(float(v) - 1000.0) < 1e-9 for v in values):
                        needs_repair = True
        except Exception:
            needs_repair = True

        if not needs_repair:
            continue

        c = cache.get(str(uid))
        if not c:
            continue

        try:
            idx = getattr(f, "index", None)
            if idx is None:
                continue
            av_series = FastSeries(index=idx, value=c["availability"], name="availability")
            price_series = FastSeries(index=idx, value=c["price_EOM"], name="price_EOM")
            rl_series = FastSeries(index=idx, value=c["residual_load_EOM"], name="residual_load_EOM")
            fuel_key = c.get("fuel_key", "fuel")
            fuel_series = FastSeries(index=idx, value=c["fuel_price"], name=f"fuel_price_{fuel_key}")

            f.availability = av_series
            f.price = {"EOM": price_series}
            f.residual_load = {"EOM": rl_series}
            f.fuel_prices = {str(fuel_key): fuel_series}

            strat = unit.bidding_strategies.get("EOM")
            if strat is not None and hasattr(strat, "prepare_observations"):
                try:
                    strat.prepare_observations(unit, "EOM")
                except Exception:
                    pass
            logger.info("Forecast re-applied for unit %s after setup.", uid)
        except Exception as e:
            logger.warning("Failed to repair forecast for unit %s: %s", uid, e)


def wait_for_forecast(powerworld, timeout: float = 60.0, poll_interval: float = 0.1):
    """
    Wait for forecast data to arrive before starting an episode.
    
    Blocks until forecast_applied_event is set or timeout is reached. Does not clear
    the event (producer clears on next update). Continues with warning if timeout exceeded.
    
    Args:
        powerworld (PowerWorld): The PowerWorld instance to wait on
        timeout (float): Maximum wait time in seconds (default: 60.0)
        poll_interval (float): Time between status checks in seconds (default: 0.1)
    
    Returns:
        None
    """
    if not hasattr(powerworld, "event_mgr"):
        logger.info("No event manager available; continuing without forecast wait.")
        return
    evt = getattr(powerworld.event_mgr, "forecast_applied_event", None)
    if evt is None:
        logger.info("No forecast_applied_event available; continuing without wait.")
        return
    if evt.is_set():
        logger.info("Forecast signal already set; continuing.")
        return
    logger.info(f"Waiting for forecast (timeout: {timeout}s)...")
    start = time.time()
    while not evt.is_set() and (time.time() - start) < timeout:
        if getattr(powerworld.event_mgr, "stop_requested", False):
            logger.info("Stop requested while waiting for forecast.")
            return
        time.sleep(poll_interval)
    if evt.is_set():
        logger.info("Forecast received; continuing.")
    else:
        logger.warning("No forecast received within timeout; continuing anyway.")


def power_run_learning(
    powerworld: PowerWorld,
    inputs_path: str,
    run_final_eval: bool = False,
) -> None:
    """
    Run the RL training process for PowerWorld agents.
    
    Orchestrates the full training loop including episode setup, policy initialization,
    validation at intervals, and optional final evaluation. Coordinates with simulation
    via event manager for synchronization. Mimics run_learning function of ASSUME.
    
    Args:
        powerworld (PowerWorld): Initialized PowerWorld instance
        inputs_path (str): Path to scenario input files
        run_final_eval (bool): Run final evaluation after training (default: False)
    
    Returns:
        None
    
    Raises:
        AssumeException: If continuing learning but policies don't exist
    """
    
    logger.info("=" * 70)
    logger.info("STARTING POWER LEARNING PROCESS")
    logger.info("=" * 70)

    temp_csv_path = powerworld.export_csv_path
    powerworld.export_csv_path = inputs_path

    powerworld.learning_role.initialize_policy(actors_and_critics=None)
    powerworld.learning_role.rl_algorithm.initialize_policy()

    save_path = powerworld.learning_config["trained_policies_save_path"]
    save_path_obj = Path(save_path)
    policies_exist = save_path_obj.is_dir() and any(save_path_obj.iterdir())
    continue_learning = powerworld.learning_config.get("continue_learning", False)

    if policies_exist:
        if continue_learning:
            logger.warning(
                f"Save path '{save_path}' exists. Continue-learning active – data may be overwritten."
            )
        else:
            logger.warning(f"Save path '{save_path}' exists –  starting fresh and overwriting.")
            shutil.rmtree(save_path, ignore_errors=True)
            os.makedirs(save_path, exist_ok=True)
    else:
        if continue_learning:
            raise AssumeException("Policy to continue learning not found.")
        os.makedirs(save_path, exist_ok=True)

    tensorboard_path = f"tensorboard/{powerworld.scenario_data['simulation_id']}"
    if os.path.exists(tensorboard_path):
        shutil.rmtree(tensorboard_path, ignore_errors=True)


    # Inter-episodische Daten
    inter_episodic_data = {
        "buffer": ReplayBuffer(
            buffer_size=int(powerworld.learning_config.get("replay_buffer_size", 5e5)),
            obs_dim=powerworld.learning_role.rl_algorithm.obs_dim,
            act_dim=powerworld.learning_role.rl_algorithm.act_dim,
            n_rl_units=len(powerworld.learning_role.rl_strats),
            device=powerworld.learning_role.device,
            float_type=powerworld.learning_role.float_type,
        ),
        "actors_and_critics": None,
        "max_eval": defaultdict(lambda: -1e9),
        "all_eval": defaultdict(list),
        "avg_all_eval": [],
        "episodes_done": 0,
        "eval_episodes_done": 0,
        "noise_scale": powerworld.learning_config.get("noise_scale", 1.0),
    }
    powerworld.learning_role.load_inter_episodic_data(inter_episodic_data)
    powerworld.learning_role.episodes_collecting_initial_experience = powerworld.learning_config[
        "episodes_collecting_initial_experience"
    ]

    training_eps = powerworld.learning_role.training_episodes
    validation_interval = min(training_eps, powerworld.learning_config.get("validation_episodes_interval", 3))

    eval_episode = 1
    broke_early = False

    for episode in range(1, training_eps + 1):

        if getattr(powerworld, "event_mgr", None) and powerworld.event_mgr.stop_requested:
            logger.info("Stop requested from API, terminating training")
            broke_early = True
            break

        powerworld.current_episode = episode
        is_eval = (
            episode % validation_interval == 0
            and training_eps > episode >= powerworld.learning_role.episodes_collecting_initial_experience
        )

        powerworld.reset()

        if is_eval:
            logger.info("-" * 60)
            logger.info(f"EVALUATION - Episode {episode} (Eval #{eval_episode})")
            logger.info("-" * 60)

            setup_world(world=powerworld, evaluation_mode=True, eval_episode=eval_episode)
        else:
            logger.info("=" * 70)
            logger.info(f"TRAINING - Episode {episode}/{training_eps}")
            logger.info("=" * 70)

            setup_world(world=powerworld, episode=episode)

        powerworld.learning_role.load_inter_episodic_data(inter_episodic_data)
        verify_policy_state(powerworld, f"Episode {episode} start")

        wait_for_forecast(powerworld)
        _repair_forecast_if_needed(powerworld)
        powerworld.run()
        powerworld.learning_role.tensor_board_logger.update_tensorboard()

        inter_episodic_data = powerworld.learning_role.get_inter_episodic_data()
        inter_episodic_data["episodes_done"] = episode

        # Rewards & Policies behandeln
        total_rewards = powerworld.current_episode_reward
        avg_reward = float(np.mean(total_rewards)) if hasattr(total_rewards, "__iter__") else float(total_rewards)

        if is_eval:
            terminate = powerworld.learning_role.compare_and_save_policies({"avg_reward": avg_reward})
            inter_episodic_data["eval_episodes_done"] = eval_episode
            eval_episode += 1
            if terminate:
                logger.info("Early stopping triggered by compare/save criterion.")
                broke_early = True
                break
        else:
            powerworld.learning_role.rl_algorithm.save_params(
                directory=f"{powerworld.learning_role.trained_policies_save_path}/last_policies"
            )
            inter_episodic_data["episodes_done"] = episode

        if episode < training_eps:
            if getattr(powerworld, "event_mgr", None) and powerworld.event_mgr.stop_requested:
                broke_early = True
                break
            wait_for_episode_signal(powerworld)

    if run_final_eval and (not broke_early):
        logger.info("=" * 70)
        logger.info("FINAL EVALUATION RUN")
        logger.info("=" * 70)

        powerworld.reset()
        setup_world(world=powerworld, evaluation_mode=True, eval_episode=eval_episode)
        powerworld.learning_role.load_inter_episodic_data(inter_episodic_data)
        powerworld.run()
        powerworld.learning_role.tensor_board_logger.update_tensorboard()
        if hasattr(powerworld, "training_state"):
            final_reward = powerworld.training_state.get("total_rewards", 0)
            logger.info("=" * 70)
            logger.info("FINAL EVALUATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"Final evaluation reward: {final_reward}")

    powerworld.export_csv_path = temp_csv_path
    powerworld.reset()

    logger.info("POWER LEARNING PROCESS COMPLETED")


def power_run_evaluation(
        powerworld: PowerWorld,
) -> None:
    """
    Evaluate a pre-trained RL agent in PowerWorld without training.
    
    Runs a single evaluation episode using pre-trained policies. Does not perform
    any policy updates. Reports evaluation results and notifies external systems
    of completion.
    
    Args:
        powerworld (PowerWorld): Initialized PowerWorld instance
        policy_path (str): Path to pre-trained policies (default: uses learning_config)
        verbose (bool): Enable verbose logging (default: False)
    
    Returns:
        None
    """
    # -----------------------------------------------------------
    # 1 - Initialization
    logger.info("=" * 70)
    logger.info("STARTING POLICY EVALUATION PROCESS")
    logger.info("=" * 70)
    logger.info("PHASE 1: INITIALIZATION")

    # check if we already stored policies for this simulation
    save_path = powerworld.learning_config["trained_policies_save_path"]
    load_path = powerworld.learning_config.get("trained_policies_load_path", save_path)
    avg_reward_path = os.path.join(load_path, "avg_reward_eval_policies")
    powerworld.learning_config["trained_policies_load_path"] = avg_reward_path
    powerworld.learning_role.trained_policies_load_path = avg_reward_path

    logger.info(f"Using pre-trained policies from: {save_path}")


    # Override learning configuration for evaluation mode
    powerworld.learning_config.update({
        "learning_mode": False,
        "continue_learning": True,
        "evaluation_mode": True,
        "trained_policies_load_path": avg_reward_path,
    })

    # -----------------------------------------------------------
    # 2 - Setup and initialize learning role


    # Setup world for evaluation
    setup_world(
        world=powerworld,
        terminate_learning=False,  # Critical flag to use pre-trained policies
        evaluation_mode=True
    )

    # Initialize the RL algorithm first to set dimensions
    powerworld.learning_role.rl_algorithm.initialize_policy()
    powerworld.learning_role.rl_algorithm.load_params(directory=avg_reward_path)

    # Extract the policy to preserve it in inter_episodic_data
    extracted_policy = powerworld.learning_role.rl_algorithm.extract_policy()

    # Create a minimal inter_episodic_data structure for the learning role
    # This is needed even though we're not training
    inter_episodic_data = {
        "buffer": ReplayBuffer(
            buffer_size=int(powerworld.learning_config.get("replay_buffer_size", 5e5)),
            obs_dim=powerworld.learning_role.rl_algorithm.obs_dim,
            act_dim=powerworld.learning_role.rl_algorithm.act_dim,
            n_rl_units=len(powerworld.learning_role.rl_strats),
            device=powerworld.learning_role.device,
            float_type=powerworld.learning_role.float_type,
        ),
        "actors_and_critics": extracted_policy,
        "max_eval": defaultdict(lambda: -1e9),
        "all_eval": defaultdict(list),
        "avg_all_eval": [],
        "episodes_done": 0,
        "eval_episodes_done": 0,
        "noise_scale": 0.0,  # No exploration noise during evaluation
    }

    # -----------------------------------------------------------
    # 3 - Run evaluation
    logger.info("=" * 70)
    logger.info("PHASE 2: RUNNING EVALUATION")
    logger.info("=" * 70)

    # Load inter-episodic data for the learning role
    powerworld.learning_role.load_inter_episodic_data(inter_episodic_data)

    policy_stats = verify_policy_state(powerworld, "Before evaluation")
    logger.info(f"Policy statistics: {policy_stats}")

    # Run the simulation
    logger.info("Starting evaluation...")
    wait_for_forecast(powerworld)
    _repair_forecast_if_needed(powerworld)
    powerworld.run()
    powerworld.learning_role.tensor_board_logger.update_tensorboard()

    # -----------------------------------------------------------
    # 4 - Report results
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    # Extract final rewards if available
    if hasattr(powerworld, 'training_state'):
        final_reward = powerworld.training_state.get("total_rewards", 0)
        logger.info(f"Evaluation reward: {final_reward}")

    # Report unit-specific rewards if available
    if "Operator-RL" in powerworld.unit_operators:
        logger.info("Unit-specific rewards:")
        for unit_id, unit in powerworld.unit_operators["Operator-RL"].units.items():
            if hasattr(unit, 'outputs') and 'reward' in unit.outputs and len(unit.outputs['reward']) > 0:
                total_reward = float(unit.outputs['reward'].sum())
                logger.info(f"  {unit_id}: {total_reward}")

    powerworld.notify_training_complete()

    logger.info("POLICY EVALUATION COMPLETED")
    logger.info("=" * 70)


def wait_for_episode_signal(powerworld):
    """
    Wait synchronously for a signal to start the next episode.
    
    Blocks execution until market simulation signals via next_episode_event that the next
    episode can begin, or until a stop request is received. Clears the event after
    receiving it.
    
    Args:
        powerworld (PowerWorld): The PowerWorld instance containing the event manager
    
    Returns:
        None
    """

    # Simple blocking loop until the event is set
    while not hasattr(powerworld, 'event_mgr') or not powerworld.event_mgr.next_episode_event.is_set():

        # Check for stop request
        if hasattr(powerworld, 'event_mgr') and powerworld.event_mgr.stop_requested:
            logger.info("Stop requested while waiting for next episode")
            return

        time.sleep(0.1)  # Simple blocking sleep
    # Reset signal for next use
    powerworld.event_mgr.next_episode_event.clear()

def verify_policy_state(powerworld, stage="unknown"):
    """
    Log and extract neural network parameter statistics for verification.
    
    Collects statistics (mean, norm, min, max) from the first and last layers of
    each unit's actor network. Used for debugging and monitoring policy changes
    during training and evaluation.
    
    Args:
        powerworld (PowerWorld): The PowerWorld instance containing the learning role
        stage (str): Description of current stage (e.g., 'Episode 5 start')
    
    Returns:
        dict: Statistics per unit with format:
            {
                unit_id: {
                    "layer_0": {"mean": float, "norm": float, "min": float, "max": float},
                    "layer_n": {...}
                }
            }
    """

    policy_stats = {}
    for unit_id, strategy in powerworld.learning_role.rl_strats.items():
        if hasattr(strategy, 'actor'):
            unit_stats = {}
            params = list(strategy.actor.parameters())
            # Sample first and last layers
            for i, param in enumerate(params):
                if i == 0 or i == len(params) - 1:  # First and last layers
                    param_tensor = param.detach().cpu()
                    unit_stats[f"layer_{i}"] = {
                        "mean": float(param_tensor.mean()),
                        "norm": float(param_tensor.norm()),
                        "min": float(param_tensor.min()),
                        "max": float(param_tensor.max())
                    }
            policy_stats[unit_id] = unit_stats

    logger.info(f"Policy state [{stage}]: {policy_stats}")

    return policy_stats