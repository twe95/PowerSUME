# PowerSUME: Simulation Reinforcement Learning Bridge for ASSUME

![Python Version](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19098073.svg)](https://doi.org/10.5281/zenodo.19098073)


# Introduction
PowerSUME serves as an interface code that makes the reinforcement‑learning capabilities of the ASSUME framework usable for other agent‑based electricity‑market models. It enables data exchange via an HTTP interface, allowing the electricity‑market model to transmit power‑plant data and forecast information to PowerSUME. PowerSUME then manages the learning environment and, using ASSUME, returns bids.

For more details about the ASSUME framework and its capabilities, please consult the project website https://assume-project.de/. Note that we currently rely on a specific commit of ASSUME (see Prerequisites) to ensure compatibility; the latest version may require adjustments.

At present, only a single price can be passed per unit, and it always markets the maximum available generation capacity.

To use the interface effectively, an additional market model is required that can communicate with the interface according to the endpoints and data contract shown below. In general we only recommend PowerSUME in case, your own model brings an advanadge over ASSUME's market simulation capabilities, for example because you have an existing model with specific features or a preferred modeling approach. If you are starting from scratch, we recommend using ASSUME's built-in market simulation features directly, which are designed to work seamlessly with the learning environment.

## Features

- FastAPI interface between market simulation and ASSUME
- Episode-based RL training loop with evaluation checkpoints
- Forecast synchronization
- Currently only single bids per unit

## Repository Layout

- `PowerSUME_api_01.py` - FastAPI application and session management
- `power_learning_01.py` - training and evaluation orchestration
- `powerworld.py` - simulation integration (ASSUME `World` extension)
- `requirements.txt` - runtime and development dependencies
- `setup.py` - packaging metadata and entry points
- `install.sh` - optional Linux/macOS setup helper

## Prerequisites

- Python 3.12
- Market model side API integration for end-to-end execution

ASSUME is intentionally pinned to this exact commit for compatibility and will be installed with the requirements:

```bash
pip install git+https://github.com/assume-framework/assume.git@0b457f6d212163ce42064f89c4b34b8a04a2e5e0#egg=assume-framework
```

## Python-Side Runtime Start

Run the API from the project root. You can also call the main function from `powersume-api`.:

```bash
python -m uvicorn PowerSUME_api_01:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## API Endpoints (Implemented)

- `GET /health`
- `POST /session/init`
- `GET /session/{session_id}/status`
- `POST /session/{session_id}/forecast`
- `GET /session/{session_id}/bids`
- `POST /session/{session_id}/orderbook`
- `POST /session/{session_id}/begin_next_day`
- `POST /session/{session_id}/begin_next_episode`
- `POST /session/{session_id}/episode_complete`
- `POST /session/{session_id}/training_complete`
- `POST /session/{session_id}/stop`

`/session/{session_id}/params` and `/session/{session_id}/forecast/array` are not implemented in the current Python API.

## Python-Side Data Contract

This section describes only what the Python API expects and produces.

### 1) Session Initialization

Endpoint: `POST /session/init`

Request body:

```json
{
  "output_directory": "C:/runs/rl-session/",
  "inputs_path": "C:/runs/rl-session",
  "scenario": "",
  "study_case": "base",
  "java_api_url": "http://127.0.0.1:8000",
  "test_mode": false,
  "verbose": false,
  "evaluation_mode": false,
  "forecastUpdateFrequency": 14
}
```

Important fields:

- `output_directory` and `inputs_path` must point to valid local paths used by ASSUME loader.
- `java_api_url` is the callback base URL used by `PowerWorld` for episode/training notifications.
- `evaluation_mode=true` runs evaluation-only flow.

Response contains `session_id` and initial status.

### Scenario Files Needed At Startup (`inputs_path`)

When `POST /session/init` is called, the Python side loads the scenario from `inputs_path` via ASSUME (`load_scenario_folder(...)`).
In practice, the integration must ensure these files exist before initialization starts. Availability, forecasts, demand
fuel prices will be overwritten by the market simulation during runtime, but the initial files must be present for the scenario to load successfully. 
The exact file requirements depend on the ASSUME scenario setup, but a typical minimal set includes:

Expected files at `inputs_path`:

- `config.yaml`
- `powerplant_units.csv`
- `demand_units.csv`
- `availability_df.csv`
- `forecasts_df.csv`
- `demand_df.csv`
- `fuel_prices_df.csv`

What each file is used for:

- `config.yaml`: base simulation and learning configuration (time range, episode length, RL algorithm settings, market config).
- `powerplant_units.csv`: RL-relevant generating units and static unit parameters.
- `demand_units.csv`: demand-side unit definitions used in market setup.
- `availability_df.csv`: time series for per-unit availability.
- `forecasts_df.csv`: time series for forecast market prices.
- `demand_df.csv`: time series for residual/system demand.
- `fuel_prices_df.csv`: time series for fuel/CO2 price inputs.

Minimal format expectations:

- CSV delimiter: comma (`,`), first row header.
- Timestamp column name: `datetime`.
- Datetime format: `YYYY-MM-DD HH:MM:SS` (or ISO-compatible string parseable by Python datetime handling).
- Time resolution: hourly, aligned across all `*_df.csv` files.
- Unit time-series columns must use the same unit IDs as defined in `powerplant_units.csv`.

Commonly used headers written by the integration:

- `powerplant_units.csv`:
  `name,technology,bidding_EOM,fuel_type,emission_factor,max_power,min_power,efficiency,additional_cost,unit_operator`
- `demand_units.csv`:
  `name,technology,bidding_EOM,max_power,min_power,unit_operator`
- `availability_df.csv`:
  `datetime,<unit_id_1>,<unit_id_2>,...`
- `forecasts_df.csv`:
  `datetime,price_EOM`
- `demand_df.csv`:
  `datetime,demand_EOM`
- `fuel_prices_df.csv`:
  `datetime,RES,co2,<fuel_1>,<fuel_2>,...`

In addition, `output_directory` should be writable. During session setup/runtime, Python creates/uses:

- `local_db/` (SQLite storage)
- `outputs/<session_id>/` (CSV outputs for the current session)

### 2) Forecast Submission

Endpoint: `POST /session/{session_id}/forecast`

Request body:

```json
{
  "units": [
	{
	  "unit_ids": [101],
	  "start_epoch": 1735689600,
	  "step_sec": 3600,
	  "len": 96,
	  "rl": [50000.0, 49000.0],
	  "av": [1.0, 0.95],
	  "vc": [20.0, 20.1],
	  "p": [80.0, 78.5]
	}
  ]
}
```

Rules enforced by Python endpoint:

- Top-level `units` must be a non-empty array.
- `unit_ids` must be an integer list (single integer is also accepted and normalized internally).
- `rl` (residual load), `av` (availability), `vc` (variable costs), `p` (price forecast) must be numeric arrays.
- Arrays are conformed to each unit forecaster index length (trimmed or padded with last value).
- `start_epoch`, `step_sec`, `len` are part of the contract and should be sent for consistency.

### 3) Bids Retrieval

Endpoint: `GET /session/{session_id}/bids`

Response while waiting:

```json
{
  "status": "waiting",
  "message": "No bids ready yet",
  "bids": [],
  "timestamp": "2026-03-18T10:00:00"
}
```

Response when ready:

```json
{
  "status": "ready",
  "bids": [
	{
	  "unit_id": 101,
	  "price": 72.4,
	  "volume": 120.0,
	  "start_time": "2026-01-01 00:00:00",
	  "end_time": "2026-01-01 01:00:00",
	  "strategy_type": "pp_learning"
	}
  ],
  "timestamp": "2026-03-18T10:00:01"
}
```

### 4) Orderbook Submission

Endpoint: `POST /session/{session_id}/orderbook`

Request body:

```json
{
  "market_prices": [
	{
	  "start_time": "2026-01-01 00:00:00",
	  "end_time": "2026-01-01 01:00:00",
	  "price": 65.0,
	  "volume_sell": 1000.0,
	  "volume_ask": 980.0,
	  "volume_traded": 970.0
	}
  ],
  "accepted_bids": [
	{
	  "unit_id": 101,
	  "start_time": "2026-01-01 00:00:00",
	  "end_time": "2026-01-01 01:00:00",
	  "price": 72.4,
	  "volume": 120.0,
	  "accepted_volume": 120.0,
	  "accepted_price": 65.0,
	  "node": "default_node"
	}
  ]
}
```

Python processing expects at minimum:

- `market_prices[]`: timestamps (used to build hour coverage).
- `accepted_bids[]`: `start_time`, `end_time`, `price`, `volume`, `accepted_volume`, `accepted_price`, `unit_id`; `node` is optional.
- Datetime strings must be parseable by Python `datetime.fromisoformat`, for example `YYYY-MM-DD HH:MM:SS`.

### 5) Day/Episode Control Signals

- `POST /session/{session_id}/begin_next_day` after orderbook acceptance.
- `POST /session/{session_id}/begin_next_episode` when status indicates `episode_complete`.
- `POST /session/{session_id}/stop` for graceful shutdown.

## Recommended Integration Sequence

1. Start API and wait for `GET /health` with `{"status": "ok"}`.
2. Create a session via `POST /session/init`.
3. Poll `GET /session/{session_id}/status` until `status` is `running`.
4. Submit forecast data via `POST /session/{session_id}/forecast`.
5. Poll `GET /session/{session_id}/bids` until `status` is `ready`.
6. Run your market clearing and submit `POST /session/{session_id}/orderbook`.
7. Trigger `POST /session/{session_id}/begin_next_day`.
8. Repeat steps 4-7; on episode boundary, call `begin_next_episode` when needed.

## Minimal Validation Checklist

- Session starts and moves to `running`.
- Forecast endpoint returns `{"status": "accepted"}`.
- Bids endpoint transitions `waiting -> ready`.
- Orderbook endpoint returns `{"status": "accepted"}`.
- Status eventually reaches `episode_complete`/`completed` depending on mode.

## Notes on Modes

- **Training mode**: policy updates enabled (`evaluation_mode=false`)
- **Evaluation mode**: deterministic inference only (`evaluation_mode=true`)

The mode is selected via the `/session/init` payload.

## Development

All dependencies are pinned to fixed versions in `requirements.txt` for reproducible setups.

## License

Copyright 2026 Karlsruhe Institute of Technology.

PowerSUME is licensed under the GNU Affero General Public License v3.0. This license is a strong copyleft license that requires that any derivative work be licensed under the same terms as the original work. It is approved by the Open Source Initiative.
