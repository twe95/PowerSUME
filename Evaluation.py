# app.py
# -*- coding: utf-8 -*-
"""
Streamlit app to analyze RL-generated orderbooks across episodes and days (ABSOLUTE PRICE BINS).

It loads JSON files named like: episode_<EPISODE>_day_<DAY>.json with keys:
- "accepted_bids": list[dict]
- "market_prices": list[dict] (optional)

IMPORTANT
---------
- We treat the unit-specific **accepted_price** as the relevant market price for that unit/time.
  This makes all market aggregates react to the selected units (unit-specific "market" line).
- If a market_prices block exists, it's only used as a fallback when accepted_price is missing.

Memory / I/O controls
---------------------
- Load only the latest N episodes (pre-filter at file-selection time).
- Optionally load only evaluation episodes: every m-th episode with **0 never evaluation**,
  i.e. episodes where (episode + 1) % m == 0 → m-1, 2m-1, 3m-1, ...

Key points
----------
- Absolute price analysis (no quantiles).
- Bin width configurable; ranges via number inputs (no sliders).
- Fast IO with polars, binning with numpy.searchsorted (version-agnostic).
- Optional auto-refresh via streamlit-autorefresh (safe fallback button).
- Charts:
  5) Line  — hourly avg market (accepted_price) + per-episode avg bid (recency-coded).
  6) Line  — hourly market (left) vs. per-episode acceptance (right, aligned grids, recency-coded).
  7) Line  — weekly (168h) avg market + per-episode avg bid (recency-coded).
  8) Line  — hourly avg market vs. **per-episode average revenue price** (recency-coded).
             Revenue price rule: if accepted_volume > 0 → revenue = accepted_price, else 0.
  8W) Line — weekly (168h) avg market vs. **per-episode average revenue price** (recency-coded).
  9) Bar+Line — per-episode avg **revenue price** & avg market (bars) with acceptance rate (secondary axis).
  10) Box — distribution of market prices per episode (box-plot by episode).
  11) Bar — count of hours with negative prices and accepted bids per episode.

All code comments in English (per user preference).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List, Optional

import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Optional auto-refresh helper (separate package)
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
except Exception:
    st_autorefresh = None  # graceful fallback

# Fallback: older streamlit versions may not have cache_data; provide a no-op decorator
if not hasattr(st, "cache_data"):
    def _cache_data_noop(*args, **kwargs):
        def _deco(f):
            return f
        return _deco

    st.cache_data = _cache_data_noop


# --------------------------
# --------- Utils ----------
# --------------------------

FILENAME_RE = re.compile(r"episode_(?P<episode>\d+)_day_(?P<day>\d+)\.json$", re.IGNORECASE)


def parse_episode_day_from_name(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Extract (episode, day) from filename."""
    m = FILENAME_RE.search(path.name)
    if not m:
        return None, None
    return int(m.group("episode")), int(m.group("day"))


def safe_read_json(path: Path) -> Dict[str, Any]:
    """Read JSON safely (return {} on error)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_lazy_frames(files: Iterable[Path]) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create LazyFrames for bids and market prices from many JSON files."""
    bids_lf_list, market_lf_list = [], []

    for p in files:
        ep, day = parse_episode_day_from_name(p)
        if ep is None or day is None:
            continue

        data = safe_read_json(p)
        if not data:
            continue

        bids = data.get("accepted_bids", [])
        # normalize bids: ensure dicts, unify marketArea key and set default 'ALL'
        norm_bids = []
        if isinstance(bids, list):
            for item in bids:
                if not isinstance(item, dict):
                    continue
                d = dict(item)  # shallow copy
                # unify possible key names
                if d.get("market_area") is not None and d.get("marketArea") is None:
                    d["marketArea"] = d.pop("market_area")
                if d.get("marketArea") is None:
                    d["marketArea"] = "ALL"
                # ensure common expected keys exist (polars will error if missing when referenced)
                for _k in ("start_time", "end_time", "unit_id", "price", "accepted_price", "accepted_volume", "volume", "node", "forecasted_price"):
                    if _k not in d:
                        d[_k] = None
                norm_bids.append(d)
        if norm_bids:
            lf_bids = (
                pl.LazyFrame(norm_bids)
                .with_columns(
                    episode=pl.lit(ep, dtype=pl.Int32),
                    day=pl.lit(day, dtype=pl.Int32),
                )
                .with_columns(
                    pl.col("start_time").str.strptime(pl.Datetime, strict=False).alias("start_time"),
                    pl.col("end_time").str.strptime(pl.Datetime, strict=False).alias("end_time"),
                    pl.col("unit_id").cast(pl.Utf8).fill_null("unknown"),
                    pl.col("price").cast(pl.Float64).alias("bid_price"),
                    pl.col("accepted_price").cast(pl.Float64).alias("accepted_price"),
                    pl.col("accepted_volume").cast(pl.Float64).alias("accepted_volume"),
                    pl.col("volume").cast(pl.Float64).alias("bid_volume"),
                    pl.col("node").cast(pl.Utf8).fill_null("default_node"),
                    pl.col("forecasted_price").cast(pl.Float64).alias("forecasted_price"),
                    # new: market area field in JSON (may be named 'marketArea') — fill missing with 'ALL'
                    pl.col("marketArea").cast(pl.Utf8).fill_null("ALL").alias("market_area"),
                )
            )
            bids_lf_list.append(lf_bids)

        mkt = data.get("market_prices", [])
        # normalize market prices: ensure dicts, unify marketArea key and set default 'ALL'
        norm_mkt = []
        if isinstance(mkt, list):
            for item in mkt:
                if not isinstance(item, dict):
                    continue
                d = dict(item)
                if d.get("market_area") is not None and d.get("marketArea") is None:
                    d["marketArea"] = d.pop("market_area")
                if d.get("marketArea") is None:
                    d["marketArea"] = "ALL"
                # ensure expected keys for market block
                for _k in ("start_time", "end_time", "price", "volume_traded", "volume_sell", "volume_ask"):
                    if _k not in d:
                        d[_k] = None
                norm_mkt.append(d)
        if norm_mkt:
            lf_mkt = (
                pl.LazyFrame(norm_mkt)
                .with_columns(
                    episode=pl.lit(ep, dtype=pl.Int32),
                    day=pl.lit(day, dtype=pl.Int32),
                )
                .with_columns(
                    pl.col("start_time").str.strptime(pl.Datetime, strict=False).alias("start_time"),
                    pl.col("end_time").str.strptime(pl.Datetime, strict=False).alias("end_time"),
                    pl.col("price").cast(pl.Float64).alias("market_price"),
                    pl.col("volume_traded").cast(pl.Float64).alias("mkt_volume_traded"),
                    pl.col("volume_sell").cast(pl.Float64).alias("mkt_volume_sell"),
                    pl.col("volume_ask").cast(pl.Float64).alias("mkt_volume_ask"),
                    # market block may also include marketArea
                    pl.col("marketArea").cast(pl.Utf8).fill_null("ALL").alias("market_area"),
                )
            )
            market_lf_list.append(lf_mkt)

    bids_lf = pl.concat(bids_lf_list, how="vertical_relaxed") if bids_lf_list else pl.LazyFrame([])
    market_lf = pl.concat(market_lf_list, how="vertical_relaxed") if market_lf_list else pl.LazyFrame([])
    return bids_lf, market_lf


@st.cache_data(show_spinner=False)
def load_and_join_cached(file_paths: List[str], signatures: List[Tuple[str, int]]) -> pl.DataFrame:
    """Load JSONs and build a unit-specific dataset.

    - The column `market_price` is taken from `accepted_price` (unit-specific).
    - If `accepted_price` is null, we fall back to `market_prices.price` joined on (episode, day, start_time).
    """
    files = [Path(p) for p in file_paths]
    bids_lf, market_lf = build_lazy_frames(files)
    if bids_lf.width == 0:
        return pl.DataFrame()

    if market_lf.width == 0:
        # No explicit market block: rely solely on accepted_price
        joined = (
            bids_lf
            .with_columns(
                pl.col("start_time").dt.hour().alias("hour"),
                pl.col("start_time").dt.weekday().alias("weekday"),
            )
            .with_columns((pl.col("weekday") * 24 + pl.col("hour")).alias("weekhour_raw"))
            .with_columns(
                pl.when((pl.col("weekhour_raw") >= 0) & (pl.col("weekhour_raw") <= 167))
                .then(pl.col("weekhour_raw"))
                .otherwise(None)
                .cast(pl.Int32)
                .alias("weekhour")
            )
            .with_columns(
                market_price=pl.col("accepted_price"),  # unit-specific market
                bid_minus_market=(pl.col("bid_price") - pl.col("accepted_price")),
                is_accepted=pl.when(pl.col("accepted_volume") > 0.0).then(1).otherwise(0),
            )
            .collect()
        )
        return joined

    # With market block: join as fallback only
    joined = (
        bids_lf.join(
            market_lf.select(
                "episode", "day", "start_time", "end_time",
                "market_price", "mkt_volume_traded", "mkt_volume_sell", "mkt_volume_ask",
            ),
            on=["episode", "day", "start_time"],
            how="left",
            suffix="_mkt",
        )
        .with_columns(
            market_price=pl.when(pl.col("accepted_price").is_not_null())
                          .then(pl.col("accepted_price"))
                          .otherwise(pl.col("market_price")),
        )
        .with_columns(
            pl.col("start_time").dt.hour().alias("hour"),
            pl.col("start_time").dt.weekday().alias("weekday"),
        )
        .with_columns((pl.col("weekday") * 24 + pl.col("hour")).alias("weekhour_raw"))
        .with_columns(
            pl.when((pl.col("weekhour_raw") >= 0) & (pl.col("weekhour_raw") <= 167))
            .then(pl.col("weekhour_raw"))
            .otherwise(None)
            .cast(pl.Int32)
            .alias("weekhour")
        )
        .with_columns(
            (pl.col("bid_price") - pl.col("market_price")).alias("bid_minus_market"),
            pl.when(pl.col("accepted_volume") > 0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_accepted"),
        )
        .collect()
    )
    return joined


# -------- Absolute binning helpers (numpy.searchsorted) --------

def _edges_from_range(min_v: float, max_v: float, bin_width: float) -> np.ndarray:
    """Build inclusive-right edges from absolute min/max and width."""
    if bin_width <= 0:
        bin_width = max((max_v - min_v) / 50.0, 1.0)
    lo = np.floor(min_v / bin_width) * bin_width
    hi = np.ceil(max_v / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width, dtype=float)
    if len(edges) < 2:
        edges = np.array([lo, lo + bin_width], dtype=float)
    return edges


def _bin_series_numpy(s: pl.Series, edges: np.ndarray, name: str) -> pl.Series:
    """Bin a numeric Polars Series to integer bin indices using numpy.searchsorted."""
    arr = s.to_numpy()
    idx = np.searchsorted(edges, arr, side="right") - 1
    invalid = (idx < 0) | (idx >= (len(edges) - 1))
    idx = idx.astype(np.int64, copy=False)
    idx[invalid] = -1
    return pl.Series(name, idx)


def _add_abs_bins(df: pl.DataFrame, col: str, bin_width: float, range_min: float, range_max: float,
                  idx_name: str, center_name: str) -> Tuple[pl.DataFrame, np.ndarray]:
    """Add absolute price bins (index + center) to df via LEFT JOIN lookup."""
    s = df.get_column(col).drop_nulls()
    col_min = float(s.min()) if s.len() > 0 else 0.0
    col_max = float(s.max()) if s.len() > 0 else 1.0
    lo = range_min if np.isfinite(range_min) else col_min
    hi = range_max if np.isfinite(range_max) else col_max
    edges = _edges_from_range(lo, hi, bin_width)

    idx = _bin_series_numpy(df.get_column(col), edges, name=idx_name)
    out = df.with_columns(pl.Series(idx_name, idx)).with_columns(
        pl.when(pl.col(idx_name) < 0).then(None).otherwise(pl.col(idx_name)).cast(pl.Int64).alias(idx_name)
    )

    centers = ((edges[:-1] + edges[1:]) * 0.5).astype(float)
    lut = pl.DataFrame(
        {idx_name: pl.Series(idx_name, np.arange(len(centers), dtype=np.int64)),
         center_name: pl.Series(center_name, centers)}
    ).with_columns(pl.col(idx_name).cast(pl.Int64))
    out = out.join(lut, on=idx_name, how="left")
    return out, edges


# --------------------------
# --- Episode line styles ---
# --------------------------

def _episodes_sorted(df_ep_hour: pl.DataFrame) -> List[int]:
    """Return all episode ids (sorted ascending)."""
    if df_ep_hour.is_empty():
        return []
    return sorted(int(e) for e in df_ep_hour.select("episode").unique().to_series().to_list())


def _episode_dash_map(episodes: List[int]) -> Dict[int, Dict[str, Any]]:
    """Map each episode to a line style based on recency."""
    n = len(episodes)
    style: Dict[int, Dict[str, Any]] = {}
    for i, ep in enumerate(episodes):
        rank_from_newest = n - 1 - i  # 0 is newest
        if rank_from_newest < 10:
            style[ep] = {"dash": "solid", "opacity": 1.0, "width": 3}
        elif rank_from_newest < 20:
            style[ep] = {"dash": "dash",  "opacity": 0.95, "width": 2}
        elif rank_from_newest < 30:
            style[ep] = {"dash": "dot",   "opacity": 0.9,  "width": 1.2}
        else:
            style[ep] = {"dash": "dot",   "opacity": 0.45, "width": 0.8}
    return style


# --------------------------
# ---- Grid completion -----
# --------------------------

def _complete_hours_per_episode(df_ep_hour: pl.DataFrame, value_col: str, fill_value: Optional[float]) -> pl.DataFrame:
    """Ensure every episode has rows for all hours 0..23; fill missing values."""
    if df_ep_hour.is_empty():
        return df_ep_hour
    eps = df_ep_hour.select("episode").unique()
    hours_df = pl.DataFrame({"hour": list(range(24))})
    grid = eps.join(hours_df, how="cross")
    out = grid.join(df_ep_hour, on=["episode", "hour"], how="left")
    if fill_value is not None:
        out = out.with_columns(pl.col(value_col).fill_null(fill_value))
    return out.sort(["episode", "hour"])


def _complete_weekhours_per_episode(df_ep_week: pl.DataFrame, value_col: str, fill_value: Optional[float]) -> pl.DataFrame:
    """Ensure every episode has rows for all weekhours 0..167; fill missing values."""
    if df_ep_week.is_empty():
        return df_ep_week
    eps = df_ep_week.select("episode").unique()
    wh_df = pl.DataFrame({"weekhour": list(range(168))})
    grid = eps.join(wh_df, how="cross")
    out = grid.join(df_ep_week, on=["episode", "weekhour"], how="left")
    if fill_value is not None:
        out = out.with_columns(pl.col(value_col).fill_null(fill_value))
    return out.sort(["episode", "weekhour"])


# --------------------------
# --------- UI -------------
# --------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed after `--` when launching Streamlit."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=str, default=".", help="Directory with JSON files.")
    parser.add_argument("--refresh-seconds", type=int, default=30,
                        help="Auto-refresh interval in seconds (requires streamlit-autorefresh).")
    args, _ = parser.parse_known_args()
    return args


def _ensure_order(a_min: int, a_max: int) -> tuple[int, int]:
    """Ensure min <= max for integer inputs."""
    if a_min > a_max:
        a_min, a_max = a_max, a_min
    return int(a_min), int(a_max)


def _ensure_order_float(a_min: float, a_max: float) -> tuple[float, float]:
    """Ensure min <= max for float inputs."""
    if a_min > a_max:
        a_min, a_max = a_max, a_min
    return float(a_min), float(a_max)


def sidebar_controls(df: pl.DataFrame) -> Dict[str, Any]:
    """Build sidebar controls (all ranges via number inputs)."""
    st.sidebar.header("Filters")

    # Units
    units = sorted([str(u) for u in df.select("unit_id").unique().to_series().to_list()])
    selected_units = st.sidebar.multiselect("Unit(s)", units, default=units)

    # Market areas
    # Read market_area column if present; default to single entry 'ALL' if missing
    if "market_area" in df.columns:
        m_areas = sorted([str(m) for m in df.select("market_area").unique().to_series().to_list()])
    else:
        m_areas = ["ALL"]
    selected_market_areas = st.sidebar.multiselect("Market area(s)", m_areas, default=m_areas)

    # keep full set for later logic
    all_market_areas = set(m_areas)

    # Determine data bounds
    if not df.is_empty():
        ep_min, ep_max = df.select(
            pl.col("episode").min().alias("episode_min"),
            pl.col("episode").max().alias("episode_max"),
        ).row(0)
        day_min, day_max = df.select(
            pl.col("day").min().alias("day_min"),
            pl.col("day").max().alias("day_max"),
        ).row(0)
        mkt_min, mkt_max = df.select(
            pl.col("market_price").min().alias("mkt_min"),
            pl.col("market_price").max().alias("mkt_max"),
        ).row(0)
        bid_min, bid_max = df.select(
            pl.col("bid_price").min().alias("bid_min"),
            pl.col("bid_price").max().alias("bid_max"),
        ).row(0)
    else:
        ep_min = ep_max = day_min = day_max = 0
        mkt_min = 0.0
        mkt_max = 1.0
        bid_min = 0.0
        bid_max = 1.0

    # Episodes (int inputs)
    st.sidebar.subheader("Episode range (filter inside loaded data)")
    ep_min_in = st.sidebar.number_input("Episode min", value=int(ep_min), step=1, format="%d")
    ep_max_in = st.sidebar.number_input("Episode max", value=int(ep_max), step=1, format="%d")
    ep_min_in, ep_max_in = _ensure_order(ep_min_in, ep_max_in)

    # Days (int inputs)
    st.sidebar.subheader("Day range")
    day_min_in = st.sidebar.number_input("Day min", value=int(day_min), step=1, format="%d")
    day_max_in = st.sidebar.number_input("Day max", value=int(day_max), step=1, format="%d")
    day_min_in, day_max_in = _ensure_order(day_min_in, day_max_in)

    # Hour-of-day (int inputs 0..23)
    st.sidebar.subheader("Hour-of-day range")
    hour_min_in = st.sidebar.number_input("Hour min (0–23)", value=0, min_value=0, max_value=23, step=1, format="%d")
    hour_max_in = st.sidebar.number_input("Hour max (0–23)", value=23, min_value=0, max_value=23, step=1, format="%d")
    hour_min_in, hour_max_in = _ensure_order(hour_min_in, hour_max_in)

    # Acceptance filter
    acc = st.sidebar.selectbox(
        "Acceptance",
        ["All", "Accepted only", "Rejected only", "Partly accepted only"],
        format_func=lambda x: x,
    )

    # Absolute price binning (width via number; ranges via number inputs)
    st.sidebar.header("Absolute price binning")
    mkt_bw = st.sidebar.number_input("Market price bin width", min_value=0.01, value=50.0, step=1.0, format="%.2f")

    st.sidebar.subheader("Market price range")
    mkt_lo_in = st.sidebar.number_input("Market min", value=float(np.floor(mkt_min)), step=1.0, format="%.2f")
    mkt_hi_in = st.sidebar.number_input("Market max", value=float(np.ceil(mkt_max)), step=1.0, format="%.2f")
    mkt_lo_in, mkt_hi_in = _ensure_order_float(mkt_lo_in, mkt_hi_in)

    bid_bw = st.sidebar.number_input("Bid price bin width", min_value=0.01, value=50.0, step=1.0, format="%.2f")

    st.sidebar.subheader("Bid price range")
    bid_lo_in = st.sidebar.number_input("Bid min", value=float(np.floor(bid_min)), step=1.0, format="%.2f")
    bid_hi_in = st.sidebar.number_input("Bid max", value=float(np.ceil(bid_max)), step=1.0, format="%.2f")
    bid_lo_in, bid_hi_in = _ensure_order_float(bid_lo_in, bid_hi_in)

    return {
        "units": set(selected_units),
        "market_areas": set(selected_market_areas),
        "all_market_areas": all_market_areas,
        "ep_min": ep_min_in,
        "ep_max": ep_max_in,
        "day_min": day_min_in,
        "day_max": day_max_in,
        "hour_min": hour_min_in,
        "hour_max": hour_max_in,
        "acceptance": acc,
        "mkt_bw": float(mkt_bw),
        "mkt_lo": float(mkt_lo_in),
        "mkt_hi": float(mkt_hi_in),
        "bid_bw": float(bid_bw),
        "bid_lo": float(bid_lo_in),
        "bid_hi": float(bid_hi_in),
    }


def apply_filters(df: pl.DataFrame, opts: Dict[str, Any]) -> pl.DataFrame:
    """Filter the dataset based on UI choices (within the already-loaded subset)."""
    if df.is_empty():
        return df

    flt = (
        (pl.col("unit_id").is_in(list(opts["units"])))
        & (pl.col("episode") >= opts["ep_min"])
        & (pl.col("episode") <= opts["ep_max"])
        & (pl.col("day") >= opts["day_min"])
        & (pl.col("day") <= opts["day_max"])
        & (pl.col("hour") >= opts["hour_min"])
        & (pl.col("hour") <= opts["hour_max"])
    )
    # Market area filter (if market_area column exists)
    if "market_area" in df.columns and opts.get("market_areas"):
        flt = flt & (pl.col("market_area").is_in(list(opts["market_areas"])))
    if opts["acceptance"] == "Accepted only":
        flt = flt & (pl.col("is_accepted") == 1)
    elif opts["acceptance"] == "Rejected only":
        flt = flt & (pl.col("is_accepted") == 0)
    elif opts["acceptance"] == "Partly accepted only":
        flt = flt & (pl.col("accepted_volume") > 0) & (pl.col("accepted_volume") != pl.col("bid_volume"))

    flt = flt & (pl.col("market_price") >= opts["mkt_lo"]) & (pl.col("market_price") <= opts["mkt_hi"])
    flt = flt & (pl.col("bid_price") >= opts["bid_lo"]) & (pl.col("bid_price") <= opts["bid_hi"])

    return df.filter(flt)


def _compute_market_mean(df: pl.DataFrame, group_cols: List[str], opts: Dict[str, Any]) -> pl.DataFrame:
    """Compute market mean per group respecting market-area logic.

    If the user has selected a strict subset of market areas (opts['market_areas'] != opts['all_market_areas']),
    compute the plain mean across rows. If all market areas are selected (no active filter), compute the mean per
    market_area first and then take the simple (unweighted) average across areas for each group.
    """
    if df.is_empty():
        return pl.DataFrame()

    if "market_area" not in df.columns:
        return df.group_by(group_cols).agg(pl.col("market_price").mean().alias("market_mean"))

    selected = opts.get("market_areas") or set()
    all_areas = opts.get("all_market_areas") or set()
    # If user selected a strict subset -> simple mean
    if selected and selected != all_areas:
        return df.group_by(group_cols).agg(pl.col("market_price").mean().alias("market_mean"))

    # No strict subset -> compute per-area mean then average across areas (unweighted)
    per_area = df.group_by(group_cols + ["market_area"]).agg(pl.col("market_price").mean().alias("mkt_per_area"))
    res = per_area.group_by(group_cols).agg(pl.col("mkt_per_area").mean().alias("market_mean"))
    return res


# --------------------------
# --------- Charts ----------
# --------------------------

def chart_price_duration_curve(df: pl.DataFrame, opts: Dict[str, Any], height: int = 600):
    """
    Price duration curve: plot average market clearing prices per (day, hour) position across all episodes in descending order.
    Averages are computed across all episodes for each unique (day, hour) position.
    Also includes forecasted prices as a third line.
    X-axis represents the sorted hour index (duration), Y-axis represents the average price across episodes.
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    # Aggregate by (day, hour): compute mean market price, mean bid price, and mean forecasted price ACROSS ALL EPISODES
    agg = (
        df.group_by(["day", "hour"])
        .agg(
            pl.col("market_price").mean().alias("market_mean"),
            pl.col("bid_price").mean().alias("bid_mean"),
            pl.col("forecasted_price").mean().alias("forecasted_mean"),
        )
        .drop_nulls()
        .sort(["day", "hour"])
    )

    if agg.is_empty():
        st.info("No price data available for price duration curve.")
        return

    # Convert to pandas for easier manipulation
    pdf = agg.to_pandas()

    # Sort by market prices in descending order
    pdf_sorted = pdf.sort_values("market_mean", ascending=False).reset_index(drop=True)

    # Create sorted arrays
    sorted_market_prices = pdf_sorted["market_mean"].values
    sorted_bid_prices = pdf_sorted["bid_mean"].values
    sorted_forecasted_prices = pdf_sorted["forecasted_mean"].values
    sorted_days = pdf_sorted["day"].values
    sorted_hours = pdf_sorted["hour"].values

    # X-axis: duration index (0 to N-1)
    duration_index = np.arange(len(sorted_market_prices))

    # Create custom data for hover
    hover_data = [f"Day{int(d)} H{int(h)}" for d, h in zip(sorted_days, sorted_hours)]

    # Create figure
    fig = go.Figure()

    # Add bid price curve first (so it's drawn behind)
    fig.add_trace(
        go.Scatter(
            x=duration_index,
            y=sorted_bid_prices,
            mode="lines",
            name="Avg Bid Price",
            line=dict(color="#EF553B", width=2),
            customdata=hover_data,
            hovertemplate="%{customdata}<br>Avg Bid Price: %{y:.2f}<extra></extra>",
        )
    )

    # Add forecasted price curve
    fig.add_trace(
        go.Scatter(
            x=duration_index,
            y=sorted_forecasted_prices,
            mode="lines",
            name="Avg Forecasted Price",
            line=dict(color="#00CC96", width=2, dash="dash"),
            customdata=hover_data,
            hovertemplate="%{customdata}<br>Avg Forecasted Price: %{y:.2f}<extra></extra>",
        )
    )

    # Add market price curve last (so it's drawn on top)
    fig.add_trace(
        go.Scatter(
            x=duration_index,
            y=sorted_market_prices,
            mode="lines",
            name="Avg Market Clearing Price",
            line=dict(color="#636EFA", width=2),
            customdata=hover_data,
            hovertemplate="%{customdata}<br>Avg Market Price: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Price Duration Curve (Descending Avg Prices across all Episodes)",
        xaxis_title="Duration Index (sorted by market price)",
        yaxis_title="Average Price (across episodes)",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=100),
        height=height,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def chart_episode_total_profit_by_episode(df: pl.DataFrame, opts: Dict[str, Any], height: int = 480):
    """
    Bar plot: total profit per episode where profit = market_price * accepted_volume.
    Also draw a line (secondary y-axis) showing the share of accepted volume
    over total bid volume per episode (in percent).
    Respects current filters (expects df already filtered by `apply_filters`).
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    # Ensure columns exist and nulls are treated as zero for numeric ops
    cols = [c for c in ("episode", "market_price", "accepted_volume", "bid_volume") if c in df.columns]
    dff = df.select(*cols).with_columns(
        pl.col("market_price").fill_null(0.0) if "market_price" in df.columns else pl.lit(0.0).alias("market_price"),
        pl.col("accepted_volume").fill_null(0.0) if "accepted_volume" in df.columns else pl.lit(0.0).alias("accepted_volume"),
        pl.col("bid_volume").fill_null(0.0) if "bid_volume" in df.columns else pl.lit(0.0).alias("bid_volume"),
    )

    # Compute profit per row and aggregate per episode
    dff = dff.with_columns((pl.col("market_price") * pl.col("accepted_volume")).alias("profit"))
    agg = dff.group_by("episode").agg(
        pl.col("profit").sum().alias("total_profit"),
        pl.col("accepted_volume").sum().alias("total_accepted_volume"),
        pl.col("bid_volume").sum().alias("total_bid_volume"),
    ).sort("episode")

    pdf = agg.to_pandas()
    if pdf.empty:
        st.info("No profit data available under current filters.")
        return

    # Compute accepted share safely (percent)
    pdf["total_bid_volume"] = pdf["total_bid_volume"].fillna(0.0)
    pdf["total_accepted_volume"] = pdf["total_accepted_volume"].fillna(0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        share_pct = np.zeros(len(pdf), dtype=float)
        mask = pdf["total_bid_volume"] > 0
        share_pct[mask] = (pdf.loc[mask, "total_accepted_volume"] / pdf.loc[mask, "total_bid_volume"]) * 100.0

    # Build figure: bars for total profit, line for accepted-share (%) on y2
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pdf["episode"],
            y=pdf["total_profit"],
            name="Total profit",
            hovertemplate="Ep %{x}<br>Total profit: %{y:.2f}<br>Total accepted vol: %{customdata:.2f}<extra></extra>",
            customdata=pdf["total_accepted_volume"],
            marker_color="#636EFA",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pdf["episode"],
            y=share_pct,
            mode="lines+markers",
            name="Accepted vol share (%)",
            yaxis="y2",
            hovertemplate="Ep %{x}<br>Accepted share: %{y:.2f}%<extra></extra>",
            line=dict(width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # Axis formatting for profit (left axis)
    y_min = float(np.nanmin(pdf["total_profit"]))
    y_max = float(np.nanmax(pdf["total_profit"]))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    elif y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    else:
        pad = 0.05 * (y_max - y_min)
        y_min, y_max = y_min - pad, y_max + pad

    layout = dict(
        title="Per-episode total profit (market_price × accepted_volume) + accepted-volume share",
        xaxis_title="Episode",
        yaxis_title="Total profit",
        yaxis=dict(range=[y_min, y_max]),
        yaxis2=dict(
            title="Accepted vol share (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        margin=dict(l=40, r=20, t=60, b=180),  # room for legend under the plot
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)


def chart_price_vs_price_density(df: pl.DataFrame, opts: Dict[str, Any], height: int = 520):
    """2D density heatmap of bid vs market price (absolute bins)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    dfx, x_edges = _add_abs_bins(
        df, "market_price", opts["mkt_bw"], opts["mkt_lo"], opts["mkt_hi"],
        idx_name="xb", center_name="x_center"
    )
    dfx, y_edges = _add_abs_bins(
        dfx, "bid_price", opts["bid_bw"], opts["bid_lo"], opts["bid_hi"],
        idx_name="yb", center_name="y_center"
    )
    binned = (
        dfx.filter((pl.col("xb").is_not_null()) & (pl.col("yb").is_not_null()))
        .group_by(["xb", "yb"])
        .len()
        .rename({"len": "count"})
    )
    if binned.is_empty():
        st.info("No observations in the selected ranges.")
        return
    agg = (
        binned
        .join(
            pl.DataFrame({
                "xb": np.arange(len(x_edges) - 1, dtype=np.int64),
                "x_center": (x_edges[:-1] + x_edges[1:]) / 2,
            }),
            on="xb",
            how="left",
        )
        .join(
            pl.DataFrame({
                "yb": np.arange(len(y_edges) - 1, dtype=np.int64),
                "y_center": (y_edges[:-1] + y_edges[1:]) / 2,
            }),
            on="yb",
            how="left",
        )
        .select(["x_center", "y_center", "count"])
        .to_pandas()
    )
    fig = px.density_heatmap(
        agg,
        x="x_center",
        y="y_center",
        z="count",
        histfunc="sum",
        labels={
            "x_center": "Market price (from accepted_price)",
            "y_center": "Bid price",
            "count": "Count",
        },
        title=f"Bid vs. Market (absolute bins) — width: market {opts['mkt_bw']}, bid {opts['bid_bw']}",
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def chart_episode_trajectory(df: pl.DataFrame, height: int = 520):
    """Per-unit episode trajectory of bid − market (median + IQR)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    agg = (
        df.group_by(["episode", "unit_id"])
        .agg(
            pl.col("bid_minus_market").median().alias("median_delta"),
            pl.col("bid_minus_market").quantile(0.25).alias("q25"),
            pl.col("bid_minus_market").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .sort(["unit_id", "episode"])
    )
    pdf = agg.to_pandas()
    fig = go.Figure()
    for unit, sub in pdf.groupby("unit_id"):
        fig.add_trace(
            go.Scatter(
                x=sub["episode"],
                y=sub["median_delta"],
                mode="lines+markers",
                name=f"Unit {unit} median",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["episode"],
                y=sub["q75"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["episode"],
                y=sub["q25"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name=f"Unit {unit} IQR",
            )
        )
    fig.update_layout(
        title="Episode trajectory: bid − market (median with IQR, absolute delta)",
        xaxis_title="Episode",
        yaxis_title="Bid − Market [price units]",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=140),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_hour_market_heatmap(df: pl.DataFrame, opts: Dict[str, Any], height: int = 520):
    """Heatmap of mean (bid − market) by hour and market-price bin."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    dfx, x_edges = _add_abs_bins(
        df, "market_price", opts["mkt_bw"], opts["mkt_lo"], opts["mkt_hi"],
        idx_name="mkt_bin", center_name="mkt_center"
    )
    agg = (
        dfx.filter(pl.col("mkt_bin").is_not_null())
        .group_by(["hour", "mkt_bin"])
        .agg(
            pl.col("bid_minus_market").mean().alias("mean_delta"),
            pl.len().alias("n"),
        )
        .join(
            pl.DataFrame({
                "mkt_bin": np.arange(len(x_edges) - 1, dtype=np.int64),
                "mkt_center": (x_edges[:-1] + x_edges[1:]) / 2,
            }),
            on="mkt_bin",
            how="left",
        )
        .select(["mkt_center", "hour", "mean_delta", "n"])
        .to_pandas()
    )
    fig = px.density_heatmap(
        agg,
        x="mkt_center",
        y="hour",
        z="mean_delta",
        histfunc="avg",
        labels={
            "mkt_center": "Market price (bin center, from accepted_price)",
            "hour": "Hour of day",
            "mean_delta": "Mean (bid−market)",
        },
        title=f"Mean (bid − market) by hour × market price (absolute bins, width {opts['mkt_bw']})",
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def chart_conditioned_distributions(df: pl.DataFrame, opts: Dict[str, Any], height: int = 520):
    """Bid vs market distributions across absolute market-price bins."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    dfx, x_edges = _add_abs_bins(
        df, "market_price", opts["mkt_bw"], opts["mkt_lo"], opts["mkt_hi"],
        idx_name="mkt_bin", center_name="mkt_center"
    )
    agg = (
        dfx.filter(pl.col("mkt_bin").is_not_null())
        .group_by("mkt_bin")
        .agg(
            pl.col("bid_price").median().alias("bid_median"),
            pl.col("bid_price").quantile(0.1).alias("bid_p10"),
            pl.col("bid_price").quantile(0.9).alias("bid_p90"),
            pl.col("market_price").median().alias("market_median"),
            pl.len().alias("n"),
        )
        .join(
            pl.DataFrame({
                "mkt_bin": np.arange(len(x_edges) - 1, dtype=np.int64),
                "mkt_center": (x_edges[:-1] + x_edges[1:]) / 2,
            }),
            on="mkt_bin",
            how="left",
        )
        .sort("mkt_center")
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["mkt_center"],
            y=agg["bid_median"],
            mode="lines+markers",
            name="Bid median",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["mkt_center"],
            y=agg["bid_p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["mkt_center"],
            y=agg["bid_p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Bid P10–P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["mkt_center"],
            y=agg["market_median"],
            mode="lines+markers",
            name="Market median",
        )
    )
    fig.update_layout(
        title=f"Bid vs. Market across absolute market-price bins (width {opts['mkt_bw']})",
        xaxis_title="Market price (bin center, from accepted_price)",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=140),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_hourly_market_and_bid_by_episode(df: pl.DataFrame, opts: Dict[str, Any], height: int = 520):
    """Hourly average market vs bids per episode (24h grid, recency-coded)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    # compute market mean per hour using market-area logic
    mkt_hour_df = _compute_market_mean(df, ["hour"], opts)
    mkt_hour = (
        mkt_hour_df.join(pl.DataFrame({"hour": list(range(24))}), on="hour", how="right")
        .sort("hour")
        .to_pandas()
    )
    bid_hour_ep = (
        df.group_by(["episode", "hour"])
        .agg(pl.col("bid_price").mean().alias("bid_mean"))
    )
    bid_hour_ep = _complete_hours_per_episode(bid_hour_ep, value_col="bid_mean", fill_value=None)
    pdf = bid_hour_ep.to_pandas()
    episodes = _episodes_sorted(bid_hour_ep)
    dash_map = _episode_dash_map(episodes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mkt_hour["hour"],
            y=mkt_hour["market_mean"],
            mode="lines+markers",
            name="Market (avg)",
            line=dict(width=4),
            hovertemplate="Hour %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )
    for ep, sub in pdf.groupby("episode"):
        ep_int = int(ep)
        style = dash_map.get(ep_int, {"dash": "dot", "opacity": 0.45, "width": 0.8})
        fig.add_trace(
            go.Scatter(
                x=sub["hour"],
                y=sub["bid_mean"],
                mode="lines+markers",
                name=f"Episode {ep_int}",
                line=dict(width=style["width"], dash=style["dash"]),
                opacity=style["opacity"],
                connectgaps=False,
                hovertemplate=f"Ep {ep_int} — Hour "+"%{x}<br>Bid avg: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Hourly average prices: market (bold) vs. bids per episode (recency-coded, full 24h grid)",
        xaxis_title="Hour of day",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=160),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_hourly_market_and_revenue_by_episode(df: pl.DataFrame, opts: Dict[str, Any], height: int = 520):
    """Hourly average market vs revenue price per episode (24h grid, recency-coded)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    mkt_hour_df = _compute_market_mean(df, ["hour"], opts)
    mkt_hour = (
        mkt_hour_df.join(pl.DataFrame({"hour": list(range(24))}), on="hour", how="right")
        .sort("hour")
        .to_pandas()
    )
    df_rev = df.with_columns(
        pl.when(pl.col("accepted_volume") > 0.0)
        .then(pl.col("accepted_price"))
        .otherwise(0.0)
        .alias("revenue_price")
    )
    rev_hour_ep = (
        df_rev.group_by(["episode", "hour"]).agg(pl.col("revenue_price").mean().alias("rev_mean"))
    )
    # Keep missing hours as null (do not introduce artificial zeros). Use hour-completion helper (0..23).
    rev_hour_ep = _complete_hours_per_episode(rev_hour_ep, value_col="rev_mean", fill_value=None)
    pdf = rev_hour_ep.to_pandas()
    episodes = _episodes_sorted(rev_hour_ep)
    dash_map = _episode_dash_map(episodes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mkt_hour["hour"],
            y=mkt_hour["market_mean"],
            mode="lines+markers",
            name="Market (avg)",
            line=dict(width=4),
            hovertemplate="Hour %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )
    for ep, sub in pdf.groupby("episode"):
        ep_int = int(ep)
        style = dash_map.get(ep_int, {"dash": "dot", "opacity": 0.45, "width": 0.8})
        fig.add_trace(
            go.Scatter(
                x=sub["hour"],
                y=sub["rev_mean"],
                mode="lines+markers",
                name=f"Episode {ep_int} revenue",
                line=dict(width=style["width"], dash=style["dash"]),
                opacity=style["opacity"],
                connectgaps=False,
                hovertemplate=f"Ep {ep_int} — Hour "+"%{x}<br>Revenue avg: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Hourly average prices: market (bold) vs. revenue per episode (recency-coded, full 24h grid)",
        xaxis_title="Hour of day",
        yaxis_title="Price / Revenue",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=160),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def _nice_tick_step(span: float, target_steps: int = 8) -> float:
    """Choose a nice tick step for a numeric axis."""
    if span <= 0 or not np.isfinite(span):
        return 1.0
    raw = span / max(target_steps, 1)
    power = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        step = m * power
        if raw <= step:
            return step
    return 10 * power


def chart_hourly_market_vs_acceptance_by_episode(
    df: pl.DataFrame,
    opts: Dict[str, Any],
    acc_range: tuple[float, float] = (0.0, 100.0),
    height: int = 900,
):
    """Hourly market (left axis) vs per-episode acceptance rate (right axis)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    mkt_hour_df = _compute_market_mean(df, ["hour"], opts)
    mkt_hour = (
        mkt_hour_df.join(pl.DataFrame({"hour": list(range(24))}), on="hour", how="right")
        .sort("hour")
        .to_pandas()
    )
    acc_hour_ep = (
        df.group_by(["episode", "hour"])
        .agg((pl.col("is_accepted").mean() * 100).alias("acc_pct"))
    )
    acc_hour_ep = _complete_hours_per_episode(acc_hour_ep, value_col="acc_pct", fill_value=0.0)
    acc_pdf = acc_hour_ep.to_pandas()

    # Left y-axis bounds
    y1_min_data = float(np.nanmin(mkt_hour["market_mean"]))
    y1_max_data = float(np.nanmax(mkt_hour["market_mean"]))
    if (
        not np.isfinite(y1_min_data)
        or not np.isfinite(y1_max_data)
        or y1_min_data == y1_max_data
    ):
        y1_min, y1_max = y1_min_data - 1.0, y1_max_data + 1.0
    else:
        pad = 0.05 * (y1_max_data - y1_min_data)
        y1_min, y1_max = y1_min_data - pad, y1_max_data + pad
    step = _nice_tick_step(y1_max - y1_min, target_steps=8)
    y1_start = np.floor(y1_min / step) * step
    y1_end = np.ceil(y1_max / step) * step
    y1_ticks = np.arange(y1_start, y1_end + 0.5 * step, step)

    # Right axis mapping
    y2_min, y2_max = acc_range
    a = (y2_max - y2_min) / (y1_end - y1_start) if (y1_end - y1_start) != 0 else 1.0
    b = y2_min - a * y1_start
    y2_ticks = a * y1_ticks + b

    episodes = _episodes_sorted(acc_hour_ep)
    dash_map = _episode_dash_map(episodes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mkt_hour["hour"],
            y=mkt_hour["market_mean"],
            mode="lines+markers",
            name="Market (avg)",
            line=dict(width=4),
            hovertemplate="Hour %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )
    for ep, sub in acc_pdf.groupby("episode"):
        ep_int = int(ep)
        style = dash_map.get(ep_int, {"dash": "dot", "opacity": 0.45, "width": 0.8})
        fig.add_trace(
            go.Scatter(
                x=sub["hour"],
                y=sub["acc_pct"],
                mode="lines+markers",
                name=f"Ep {ep_int} acceptance",
                line=dict(width=style["width"], dash=style["dash"]),
                opacity=style["opacity"],
                hovertemplate=f"Ep {ep_int} — Hour "+"%{x}<br>Acceptance: %{y:.1f}%<extra></extra>",
                yaxis="y2",
            )
        )
    fig.update_layout(
        title=(
            "Hourly: market (left) vs. per-episode acceptance (right) "
            "— aligned grids, recency-coded, full 24h grid"
        ),
        hovermode="x unified",
        xaxis=dict(title="Hour of day"),
        yaxis=dict(
            title="Market price",
            range=[y1_start, y1_end],
            tickmode="array",
            tickvals=y1_ticks,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis2=dict(
            title="Acceptance ratio (%)",
            overlaying="y",
            side="right",
            range=[y2_min, y2_max],
            tickmode="array",
            tickvals=y2_ticks,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linecolor="rgba(0,0,0,0.25)",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=180),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def chart_weekly_market_and_bid_by_episode(df: pl.DataFrame, opts: Dict[str, Any], height: int = 900):
    """Weekly (168h) market vs bids per episode (full 168h grid, recency-coded)."""
    if df.is_empty():
        st.info("No data for current filter.")
        return
    dff = df.filter(pl.col("weekhour").is_not_null())
    mkt_week_df = _compute_market_mean(dff, ["weekhour"], opts)
    mkt_week = (
        mkt_week_df.join(pl.DataFrame({"weekhour": list(range(168))}), on="weekhour", how="right")
        .sort("weekhour")
        .to_pandas()
    )
    bid_week_ep = (
        dff.group_by(["episode", "weekhour"])
        .agg(pl.col("bid_price").mean().alias("bid_mean"))
    )
    bid_week_ep = _complete_weekhours_per_episode(bid_week_ep, value_col="bid_mean", fill_value=None)
    pdf = bid_week_ep.to_pandas()

    episodes = _episodes_sorted(bid_week_ep)
    dash_map = _episode_dash_map(episodes)
    tickpos = list(range(0, 168, 24))
    ticktext = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mkt_week["weekhour"],
            y=mkt_week["market_mean"],
            mode="lines+markers",
            name="Market (avg, weekly)",
            line=dict(width=4),
            hovertemplate="Weekhour %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )
    for ep, sub in pdf.groupby("episode"):
        ep_int = int(ep)
        style = dash_map.get(ep_int, {"dash": "dot", "opacity": 0.45, "width": 0.8})
        fig.add_trace(
            go.Scatter(
                x=sub["weekhour"],
                y=sub["bid_mean"],
                mode="lines+markers",
                name=f"Episode {ep_int}",
                line=dict(width=style["width"], dash=style["dash"]),
                opacity=style["opacity"],
                connectgaps=False,
                hovertemplate=f"Ep {ep_int} — Weekhour "+"%{x}<br>Bid avg: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=(
            "Weekly average prices (168h): market (bold) vs. bids per episode "
            "(recency-coded, full 168h grid)"
        ),
        xaxis=dict(
            title="Hour of week (0..167)",
            range=[0, 167],
            tickmode="array",
            tickvals=tickpos,
            ticktext=ticktext,
            dtick=24,
        ),
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=160),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ----- 8W: Weekly revenue vs market -----
# --------------------------

def chart_weekly_market_and_revenue_by_episode(df: pl.DataFrame, opts: Dict[str, Any], height: int = 900):
    """Weekly (168h) market average vs. per-episode average revenue price.

    Robustness: ensure we always display a marker when data exists for that weekhour by
    (1) joining to a full 0..167 grid, and (2) filling missing market_mean values from a
    simple per-weekhour fallback computed directly from `market_price` where available.
    Revenue traces keep gaps (connectgaps=False) so missing hours are not interpolated.
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    dff = df.filter(pl.col("weekhour").is_not_null())

    # Market weekly mean (may have nulls after join)
    mkt_week_df = _compute_market_mean(dff, ["weekhour"], opts)
    mkt_week = (
        mkt_week_df.join(pl.DataFrame({"weekhour": list(range(168))}), on="weekhour", how="right")
        .sort("weekhour")
        .to_pandas()
    )

    # Fallback: per-weekhour mean from raw market_price (if available)
    try:
        fb = (
            dff.group_by("weekhour").agg(pl.col("market_price").mean().alias("mkt_fallback")).collect()
        )
        if not fb.is_empty():
            fb_pdf = fb.sort("weekhour").to_pandas()
            fb_map = dict(zip(fb_pdf["weekhour"].to_list(), fb_pdf["mkt_fallback"].to_list()))
            # Fill only where market_mean is missing
            if "market_mean" in mkt_week.columns:
                mkt_week["market_mean"] = mkt_week.apply(
                    lambda r: fb_map.get(int(r["weekhour"]), r["market_mean"]) if pd.isna(r["market_mean"]) else r["market_mean"],
                    axis=1,
                )
    except Exception:
        # best-effort: leave NaNs if fallback fails
        pass

    # Revenue per episode × weekhour
    df_rev = dff.with_columns(
        pl.when(pl.col("accepted_volume") > 0.0).then(pl.col("accepted_price")).otherwise(0.0).alias("revenue_price")
    )
    rev_week_ep = df_rev.group_by(["episode", "weekhour"]).agg(pl.col("revenue_price").mean().alias("rev_mean"))
    rev_week_ep = _complete_weekhours_per_episode(rev_week_ep, value_col="rev_mean", fill_value=None)
    pdf = rev_week_ep.to_pandas()

    episodes = _episodes_sorted(rev_week_ep)
    dash_map = _episode_dash_map(episodes)

    tickpos = list(range(0, 168, 24))
    ticktext = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure()
    # Market trace: visible markers and no gap-connecting
    fig.add_trace(
        go.Scatter(
            x=mkt_week["weekhour"],
            y=mkt_week.get("market_mean"),
            mode="lines+markers",
            name="Market (avg, weekly)",
            line=dict(width=3),
            marker=dict(size=6),
            connectgaps=False,
            hovertemplate="Weekhour %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )

    # Episode revenue traces — keep gaps to avoid artificial lines
    for ep, sub in pdf.groupby("episode"):
        ep_int = int(ep)
        style = dash_map.get(ep_int, {"dash": "dot", "opacity": 0.45, "width": 0.8})
        fig.add_trace(
            go.Scatter(
                x=sub["weekhour"],
                y=sub["rev_mean"],
                mode="lines+markers",
                name=f"Episode {ep_int} revenue",
                line=dict(width=style["width"], dash=style["dash"]),
                marker=dict(size=5),
                opacity=style["opacity"],
                connectgaps=False,
                hovertemplate=f"Ep {ep_int} — Weekhour " + "%{x}<br>Revenue avg: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=(
            "Weekly average prices (168h): market (bold) vs. revenue per episode "
            "(recency-coded, full 168h grid)"
        ),
        xaxis=dict(
            title="Hour of week (0..167)",
            range=[0, 167],
            tickmode="array",
            tickvals=tickpos,
            ticktext=ticktext,
            dtick=24,
        ),
        yaxis_title="Price / Revenue",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=160),
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ----- 9: Episode bars (revenue/market) + acceptance line -----
# --------------------------

def chart_episode_bars_revenue_market_with_acceptance(df: pl.DataFrame, opts: Dict[str, Any], height: int = 720):
    """Grouped bars per episode for avg revenue price and avg market price,
    with acceptance rate (0–100 %) as a line on a secondary y-axis.
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    # Compute revenue price per observation
    df_rev = df.with_columns(
        pl.when(pl.col("accepted_volume") > 0.0)
        .then(pl.col("accepted_price"))
        .otherwise(0.0)
        .alias("revenue_price")
    )
    # ensure is_accepted exists (some upstream datasets may not include it)
    # use .columns to get names (get_columns may return Series objects and break 'in' checks)
    if "is_accepted" not in df_rev.columns:
        df_rev = df_rev.with_columns(
            pl.when(pl.col("accepted_volume") > 0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_accepted")
        )

    # compute revenue mean per episode
    agg_rev = df_rev.group_by("episode").agg(pl.col("revenue_price").mean().alias("rev_mean"))

    # compute market mean per episode using helper (honors market_area logic)
    mkt_episode_df = _compute_market_mean(df_rev, ["episode"], opts)

    # compute acceptance pct per episode
    acc_df = df_rev.group_by("episode").agg((pl.col("is_accepted").mean() * 100).alias("acc_pct"))

    # join everything together
    joined = agg_rev.join(mkt_episode_df, on="episode", how="left").join(acc_df, on="episode", how="left")
    agg = joined.sort("episode").to_pandas()

    # Ensure 'market_mean' column exists (computed by helper). If missing, fill with NaN so plotting keeps working.
    if "market_mean" not in agg.columns:
        agg["market_mean"] = np.nan

    if agg.empty:
        st.info("No data after aggregation.")
        return

    # Left axis bounds with padding
    y1_vals = np.concatenate([agg["rev_mean"].to_numpy(), agg["market_mean"].to_numpy()])
    y1_min = float(np.nanmin(y1_vals))
    y1_max = float(np.nanmax(y1_vals))
    if not np.isfinite(y1_min) or not np.isfinite(y1_max) or y1_min == y1_max:
        y1_min, y1_max = y1_min - 1.0, y1_max + 1.0
    else:
        pad = 0.05 * (y1_max - y1_min)
        y1_min, y1_max = y1_min - pad, y1_max + pad

    fig = go.Figure()

    # Bars: average revenue
    fig.add_trace(
        go.Bar(
            x=agg["episode"],
            y=agg["rev_mean"],
            name="Avg Revenue Price",
            hovertemplate="Ep %{x}<br>Revenue avg: %{y:.2f}<extra></extra>",
        )
    )

    # Bars: average market
    fig.add_trace(
        go.Bar(
            x=agg["episode"],
            y=agg["market_mean"],
            name="Avg Market Price",
            hovertemplate="Ep %{x}<br>Market avg: %{y:.2f}<extra></extra>",
        )
    )

    # Line: acceptance rate
    fig.add_trace(
        go.Scatter(
            x=agg["episode"],
            y=agg["acc_pct"],
            name="Acceptance (%)",
            mode="lines+markers",
            yaxis="y2",
            hovertemplate="Ep %{x}<br>Acceptance: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Per-episode: avg revenue & market (bars) with acceptance rate (line, right axis)",
        xaxis_title="Episode",
        yaxis=dict(title="Price", range=[y1_min, y1_max]),
        yaxis2=dict(
            title="Acceptance (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        barmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=140),
        height=height,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ----- 10: Box-plot per episode for market prices -----
# --------------------------

def chart_episode_boxplot_market_price(df: pl.DataFrame, height: int = 720):
    """Box-plot of market prices per episode.

    Each episode on the x-axis, box summarizing distribution of `market_price`
    across all (filtered) observations in that episode.
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    dff = (
        df.select("episode", "market_price")
        .drop_nulls("market_price")
        .sort("episode")
    )
    if dff.is_empty():
        st.info("No market prices available for box-plot.")
        return

    pdf = dff.to_pandas()

    fig = px.box(
        pdf,
        x="episode",
        y="market_price",
        points="outliers",
        labels={
            "episode": "Episode",
            "market_price": "Market price",
        },
        title="Distribution of market prices per episode (box-plot)",
    )
    fig.update_layout(
        height=height,
        xaxis_title="Episode",
        yaxis_title="Market price",
        margin=dict(l=40, r=20, t=60, b=140),
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ----- 11: Negative-price accepted hours per episode -----
# --------------------------
def chart_episode_negative_price_hours(df: pl.DataFrame, height: int = 720):
    """Bar chart per episode (hour-counting, de-duplicated by (episode, day, hour)).

    Counting rule:
    - Each (episode, day, hour) is counted at most once per metric, regardless of how many units/bids exist.
    Metrics:
    1) unaccepted_hours: hours with at least one unaccepted bid (accepted_volume == 0)
    2) neg_acc_hours: hours with negative market_price AND at least one accepted bid (accepted_volume > 0)
    3) neg_hours_total: hours with negative market_price (regardless of acceptance)
    """

    if df.is_empty():
        st.info("No data for current filter.")
        return

    # ---- Part A: negative-price accepted hours (episode, day, hour) ----
    df_neg_acc_hour = (
        df.select("episode", "day", "hour", "market_price", "accepted_volume")
        .with_columns(
            ((pl.col("market_price") < 0.0) & (pl.col("accepted_volume") > 0.0)).alias("neg_acc_flag")
        )
        .group_by(["episode", "day", "hour"])  # de-duplicate hour cells
        .agg(pl.col("neg_acc_flag").max().alias("neg_acc_hour_flag"))
    )

    ep_neg_acc_hours = (
        df_neg_acc_hour
        .group_by("episode")
        .agg(pl.col("neg_acc_hour_flag").sum().alias("neg_acc_hours"))
        .sort("episode")
    )

    # ---- Part B: total hours with negative market prices (episode, day, hour) ----
    df_neg_hour = (
        df.select("episode", "day", "hour", "market_price")
        .with_columns((pl.col("market_price") < 0.0).alias("neg_flag"))
        .group_by(["episode", "day", "hour"])  # de-duplicate hour cells
        .agg(pl.col("neg_flag").max().alias("neg_hour_flag"))
    )

    ep_neg_hours_total = (
        df_neg_hour
        .group_by("episode")
        .agg(pl.col("neg_hour_flag").sum().alias("neg_hours_total"))
        .sort("episode")
    )

    # ---- Part C: hours with at least one unaccepted bid (episode, day, hour) ----
    # IMPORTANT: count HOURS, not bids -> de-duplicate by (episode, day, hour)
    df_unacc_hour = (
        df.select("episode", "day", "hour", "accepted_volume")
        .with_columns((pl.col("accepted_volume") == 0.0).alias("unacc_flag"))
        .group_by(["episode", "day", "hour"])  # de-duplicate hour cells
        .agg(pl.col("unacc_flag").max().alias("unacc_hour_flag"))
    )

    ep_unaccepted_hours = (
        df_unacc_hour
        .group_by("episode")
        .agg(pl.col("unacc_hour_flag").sum().alias("unaccepted_hours"))
        .sort("episode")
    )

    # ---- Merge episode-level aggregates ----
    ep_agg = (
        ep_unaccepted_hours
        .join(ep_neg_hours_total, on="episode", how="left")
        .join(ep_neg_acc_hours, on="episode", how="left")
        .fill_null(0)
    )

    pdf = ep_agg.to_pandas()
    if pdf.empty:
        st.info("No relevant data found.")
        return

    # Dense x positions to avoid gaps when episode IDs are sparse/large
    n_eps = len(pdf)
    eps_idx = np.arange(n_eps, dtype=float)
    off = 0.20
    x_left = eps_idx - off
    x_right = eps_idx + off

    fig = go.Figure()

    # Left stacked: unaccepted HOURS (bottom)
    fig.add_trace(
        go.Bar(
            x=x_left,
            y=pdf["unaccepted_hours"],
            width=0.36,
            name="Hours with ≥1 unaccepted bid",
            marker_color="#636EFA",
            hovertemplate="Ep %{customdata}<br>Unaccepted hours: %{y}<extra></extra>",
            customdata=pdf["episode"],
        )
    )

    # Left stacked: negative & accepted HOURS (top)
    fig.add_trace(
        go.Bar(
            x=x_left,
            y=pdf["neg_acc_hours"],
            width=0.36,
            name="Hours with neg. price & ≥1 accepted bid",
            marker_color="#EF553B",
            hovertemplate="Ep %{customdata}<br>Neg & accepted hours: %{y}<extra></extra>",
            customdata=pdf["episode"],
        )
    )

    # Right single: total negative-price HOURS
    fig.add_trace(
        go.Bar(
            x=x_right,
            y=pdf["neg_hours_total"],
            width=0.36,
            name="Hours with negative market price (total)",
            marker_color="#00CC96",
            hovertemplate="Ep %{customdata}<br>Negative hours (total): %{y}<extra></extra>",
            customdata=pdf["episode"],
        )
    )

    fig.update_layout(
        title="Per-episode (hour-counting): Unaccepted (bottom) + Neg&Accepted (stacked) — vs. Negative hours (total)",
        xaxis=dict(
            tickmode="array",
            tickvals=eps_idx,
            ticktext=[str(x) for x in pdf["episode"].to_list()],
            title="Episode",
        ),
        yaxis_title="Count (hours)",
        barmode="stack",
        bargap=0.12,
        bargroupgap=0.06,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=140),
        height=height,
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ----- 12: Partly accepted bids per episode -----
# --------------------------
def chart_episode_partly_accepted_bids(df: pl.DataFrame, opts: Dict[str, Any], height: int = 480):
    """Bar plot: count per episode of bids that are partly accepted.

    A bid is considered "partly accepted" when accepted_volume > 0 and
    accepted_volume != original bid volume (field may be 'bid_volume' or 'volume').
    """
    if df.is_empty():
        st.info("No data for current filter.")
        return

    # Determine which column holds the original bid volume
    if "bid_volume" in df.columns:
        vol_col = "bid_volume"
    elif "volume" in df.columns:
        vol_col = "volume"
    else:
        st.info("No bid volume column found (expected 'bid_volume' or 'volume').")
        return

    # Select relevant columns and compute the condition
    dff = df.select("episode", "accepted_volume", vol_col)

    cond = (pl.col("accepted_volume") > 0.0) & (pl.col("accepted_volume") != pl.col(vol_col))
    partly = (
        dff.filter(cond)
        .group_by("episode")
        .agg(pl.count().alias("partly_accepted_bids"))
        .sort("episode")
    )

    total = (
        df.select("episode")
        .group_by("episode")
        .agg(pl.count().alias("total_bids"))
        .sort("episode")
    )

    joined = total.join(partly, on="episode", how="left").fill_null(0)
    pdf = joined.to_pandas()
    if pdf.empty:
        st.info("No partly-accepted bids under current filters.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pdf["episode"],
            y=pdf["partly_accepted_bids"],
            name="Partly accepted bids",
            hovertemplate="Ep %{x}<br>Partly accepted bids: %{y}<extra></extra>",
            marker_color="#EF553B",
        )
    )

    # Optionally show percentage as a line
    try:
        pct = (pdf["partly_accepted_bids"] / pdf["total_bids"]).fillna(0.0) * 100.0
        fig.add_trace(
            go.Scatter(
                x=pdf["episode"],
                y=pct,
                mode="lines+markers",
                name="Share (%)",
                yaxis="y2",
                hovertemplate="Ep %{x}<br>Share: %{y:.1f}%<extra></extra>",
                marker=dict(size=6),
                line=dict(width=2, dash="dot"),
            )
        )
        y2 = True
    except Exception:
        y2 = False

    layout = dict(
        title="Per-episode: partly accepted bids (0 < accepted_volume != bid volume)",
        xaxis_title="Episode",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=60, b=180),  # more bottom space for legend
        height=height,
        hovermode="x unified",
    )
    if y2:
        layout["yaxis2"] = dict(
            title="Share (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        )
    # place legend horizontally under the plot, centered
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# --------- Main -----------
# --------------------------

def _preselect_files_with_limits(
    all_files: List[Path],
    max_latest_episodes: int,
    eval_only: bool,
    eval_every_m: int,
) -> List[Path]:
    """Select a subset of files by episode, BEFORE loading JSON, to reduce memory.

    Strategy:
    - Extract episode numbers from filenames.
    - If eval_only: keep only episodes where **(ep + 1) % m == 0** (0 is never evaluation).
    - Keep only the latest `max_latest_episodes` episode ids (by numeric ep).
    - Return only files belonging to those selected episodes.
    """
    ep_to_files: Dict[int, List[Path]] = {}
    for p in all_files:
        ep, _ = parse_episode_day_from_name(p)
        if ep is None:
            continue
        ep_to_files.setdefault(ep, []).append(p)

    episode_ids = sorted(ep_to_files.keys())

    if eval_only and eval_every_m > 0:
        episode_ids = [ep for ep in episode_ids if ((ep + 1) % eval_every_m) == 0]

    if max_latest_episodes > 0 and len(episode_ids) > max_latest_episodes:
        episode_ids = episode_ids[-max_latest_episodes:]

    selected_files: List[Path] = []
    for ep in episode_ids:
        selected_files.extend(sorted(ep_to_files[ep], key=lambda p: p.stat().st_mtime))
    return selected_files


def main():
    """Main Streamlit entrypoint."""
    args = parse_args()
    st.set_page_config(page_title="RL Orderbook Analysis (Absolute Prices)", layout="wide")

    # Sidebar: auto-refresh
    st.sidebar.caption("Auto-refresh")
    refresh_seconds = st.sidebar.slider("Refresh seconds", 5, 120, args.refresh_seconds)
    if st_autorefresh is None:
        st.sidebar.info(
            "Auto-refresh helper not installed.\n"
            "Install with: pip install streamlit-autorefresh\n"
            "You can still click the 'Refresh now' button below."
        )
    else:
        st_autorefresh(interval=refresh_seconds * 1000, key="auto-refresh-token")

    st.title("RL Orderbook Analysis — Absolute Price Evaluation (market = accepted_price)")

    # ----- I/O-level episode limits (memory-friendly) -----
    st.sidebar.header("Episode loading (I/O level)")
    max_latest = st.sidebar.number_input(
        "Max latest episodes to load (N)", min_value=1, value=100, step=8, format="%d"
    )
    eval_only = st.sidebar.checkbox("Load only evaluation episodes (every m)", value=False)
    eval_m = st.sidebar.number_input(
        "Evaluation step m", min_value=1, value=8, step=1, format="%d"
    )
    st.sidebar.caption("")

    # Determine files (sorted by mtime for stable behavior)
    data_path = Path(args.data_dir)
    all_files_all = sorted(
        [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"],
        key=lambda p: p.stat().st_mtime,
    )

    # Preselect a reduced set of files based on latest N and optional eval filter
    selected_files = _preselect_files_with_limits(
        all_files_all,
        max_latest_episodes=int(max_latest),
        eval_only=bool(eval_only),
        eval_every_m=int(eval_m),
    )

    if not selected_files:
        st.error(
            "No files selected by the current I/O limits. "
            "Adjust 'Max latest episodes' and/or 'Evaluation step m'."
        )
        st.stop()

    # Show a concise summary of what will be loaded
    sel_eps = sorted(
        {
            parse_episode_day_from_name(p)[0]
            for p in selected_files
            if parse_episode_day_from_name(p)[0] is not None
        }
    )
    ep_span = f"{sel_eps[0]}–{sel_eps[-1]}" if sel_eps else "n/a"
    st.caption(
        f"Loading {len(selected_files)} files from {len(sel_eps)} episode(s): {ep_span} "
        f"{'(eval only)' if eval_only else ''}"
    )

    file_paths = [str(p) for p in selected_files]
    signatures = [(p.name, int(p.stat().st_mtime)) for p in selected_files]

    # Manual refresh
    if st.button("Refresh now"):
        st.rerun()

    with st.spinner("Loading & joining data..."):
        df = load_and_join_cached(file_paths, signatures)

    if df.is_empty():
        st.warning("No valid data found after loading. Check filters or file contents.")
        st.stop()

    # Sidebar filters & bin settings (number inputs)
    opts = sidebar_controls(df)

    # Plot height
    line_h = st.sidebar.slider("Line plot height (px)", 400, 1600, 900, 50)

    dff = apply_filters(df, opts)

    # KPIs
    kpi = (
        dff.select(
            pl.len().alias("rows"),
            pl.col("episode").n_unique().alias("episodes"),
            pl.col("day").n_unique().alias("days"),
            pl.col("unit_id").n_unique().alias("units"),
            (pl.col("is_accepted").mean() * 100).alias("accept_rate_pct"),
            pl.col("bid_minus_market").median().alias("median_delta"),
        )
        .fill_null(0)
        .row(0, named=True)
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{int(kpi['rows']):,}")
    c2.metric("Episodes", f"{int(kpi['episodes'])}")
    c3.metric("Days", f"{int(kpi['days'])}")
    c4.metric("Units", f"{int(kpi['units'])}")
    c5.metric("Accept rate", f"{kpi['accept_rate_pct']:.1f}%")
    c6.metric("Median (bid−market)", f"{kpi['median_delta']:.2f}")

    # Price duration curve (first plot)
    st.subheader("1) Price Duration Curve (Descending Order)")
    chart_price_duration_curve(dff, opts, height=line_h)

    # Per-episode lines (recency-coded) with full grids
    st.subheader("5) Hourly average prices: market vs. bids per episode")
    chart_hourly_market_and_bid_by_episode(dff, opts, height=line_h)

    st.subheader("8) Hourly average prices: market vs. revenue per episode")
    chart_hourly_market_and_revenue_by_episode(dff, opts, height=line_h)

    st.subheader("6) Hourly: market vs. per-episode acceptance (aligned grids)")
    chart_hourly_market_vs_acceptance_by_episode(dff, opts, acc_range=(0, 100), height=line_h)

    # Weekly 168h plot (full grid)
    st.subheader("7) Weekly average prices (168h): market vs. bids per episode")
    chart_weekly_market_and_bid_by_episode(dff, opts, height=line_h)

    # Weekly revenue plot (168h), counterpart to plot 8
    st.subheader("8W) Weekly average prices (168h): market vs. revenue per episode")
    chart_weekly_market_and_revenue_by_episode(dff, opts, height=line_h)

    # Episode-level bars (avg revenue & market) + acceptance line
    st.subheader("9) Per-episode averages: revenue & market (bars) + acceptance (line)")
    chart_episode_bars_revenue_market_with_acceptance(dff, opts, height=line_h)

    # Total profit by episode
    st.subheader("Total profit per episode")
    chart_episode_total_profit_by_episode(dff, opts, height=line_h)

    # Negative-price accepted hours per episode
    st.subheader("11) Hours with negative prices and accepted bids per episode")
    chart_episode_negative_price_hours(dff, height=line_h)

    # Box-plot for market prices per episode
    st.subheader("10) Distribution of market prices per episode (box-plot)")
    chart_episode_boxplot_market_price(dff, height=line_h)

    # Partly accepted bids per episode
    st.subheader("12) Count of partly accepted bids per episode")
    chart_episode_partly_accepted_bids(dff, opts, height=line_h)


if __name__ == "__main__":
    main()
