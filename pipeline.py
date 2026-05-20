"""
Data pipeline: ingest → join → filter → IV solve → Greeks → aggregate.

Handles both OPRA.PILLAR (equities) and GLBX.MDP3 (futures) datasets.
"""

import math
import time
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False

from config import (
    ProductConfig, PRODUCTS, FilterConfig, FILTER_DEFAULTS,
    PricingModel, get_databento_key,
)
from models import compute_greeks, solve_iv
from aggregation import build_all_outputs

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# UNDERLYING PRICE INFERENCE (Put-Call Parity)
# ══════════════════════════════════════════════════════════════

def infer_underlying_price(
    chain: pd.DataFrame,
    snapshot_date: date,
    product: ProductConfig,
    risk_free_rate: float,
) -> float:
    """
    Infer the underlying spot/futures price from the option chain via put-call parity.

    For BSM products: forward F = K* + C(K*) - P(K*) at the ATM strike K*
    where |C - P| is minimal on the nearest expiry. Spot S = F · exp(-(r-q)·T).

    For Black-76 products: futures F = K* + (C - P) · exp(r·T) at ATM of nearest
    expiry that has matched call/put pairs.

    Raises ValueError if no call/put pair can be found.
    """
    df = chain.copy()
    if "bid" in df.columns and "ask" in df.columns:
        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    if "expiry" not in df.columns:
        raise ValueError("Chain has no 'expiry' column for parity inference")

    df = df.copy()
    df["dte"] = (df["expiry"] - pd.Timestamp(snapshot_date)).dt.days
    df = df[df["dte"] >= 1]
    if df.empty:
        raise ValueError("No contracts with DTE >= 1 for parity inference")

    q = product.dividend_yield if product.dividend_yield is not None else 0.0
    r = risk_free_rate
    is_bsm = product.pricing_model == PricingModel.BSM

    for dte_val in sorted(df["dte"].unique()):
        rows = df[df["dte"] == dte_val]
        calls = rows[rows["is_call"]][["strike", "mid_price"]].rename(
            columns={"mid_price": "call_mid"}
        )
        puts = rows[~rows["is_call"]][["strike", "mid_price"]].rename(
            columns={"mid_price": "put_mid"}
        )
        paired = calls.merge(puts, on="strike")
        if paired.empty:
            continue

        paired["diff"] = (paired["call_mid"] - paired["put_mid"]).abs()
        atm = paired.loc[paired["diff"].idxmin()]
        K_star = float(atm["strike"])
        C = float(atm["call_mid"])
        P = float(atm["put_mid"])
        T = dte_val / 365.0

        if is_bsm:
            F = K_star + C - P
            S = F * math.exp(-(r - q) * T)
            logger.info(
                f"  Parity (BSM): K*={K_star:.2f}, C={C:.3f}, P={P:.3f}, "
                f"T={T:.4f}y → F={F:.3f}, S≈{S:.3f}"
            )
            return float(S)
        else:
            F = K_star + (C - P) * math.exp(r * T)
            logger.info(
                f"  Parity (Black-76): K*={K_star:.2f}, C={C:.3f}, P={P:.3f}, "
                f"T={T:.4f}y → F={F:.3f}"
            )
            return float(F)

    raise ValueError(
        f"Cannot infer underlying for {product.symbol}: no call/put pairs on any expiry"
    )


# ══════════════════════════════════════════════════════════════
# DATA INGESTION
# ══════════════════════════════════════════════════════════════

def _build_client():
    """Build authenticated Databento Historical client."""
    if not HAS_DATABENTO:
        raise ImportError(
            "databento package is not installed. "
            "Install with: pip install databento"
        )
    return db.Historical(get_databento_key())


# ── Dataset-range resolution ─────────────────────────────────
# Databento historical feeds lag real-time and don't fully materialize
# until T+1. The scheduler builds query windows from "today" without
# checking what's actually published, which produces 422 errors of the
# form `data_start_after_available_end` (today not yet there) or
# `data_end_after_available_end` (today partially there, but the
# day-after end we asked for is past the feed's high-water mark).
# These helpers query metadata.get_dataset_range to learn the real
# window and clamp queries — falling back to the most recent published
# day for the snapshot when today isn't there yet.

def _get_dataset_available_end(client, dataset: str) -> Optional[datetime]:
    """
    Query Databento for the high-water-mark of published data on `dataset`.
    Returns a UTC-aware datetime, or None if metadata cannot be fetched.
    """
    try:
        rng = client.metadata.get_dataset_range(dataset=dataset)
    except Exception as e:
        logger.warning(f"  get_dataset_range failed for {dataset}: {e}; skipping clamp")
        return None
    end_str = rng.get("end") or rng.get("end_date")
    if not end_str:
        return None
    ts = pd.Timestamp(end_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _to_utc(ts_str: str) -> pd.Timestamp:
    """Treat naive timestamps as UTC; convert tz-aware ones to UTC."""
    ts = pd.Timestamp(ts_str)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def resolve_query_window(
    avail_end: Optional[datetime],
    dataset: str,
    snapshot_date: date,
    snapshot_time: str,
    snapshot_window_seconds: int,
) -> dict:
    """
    Build per-query start/end timestamps that fit inside the dataset's
    available range.

    1. If snapshot_date is past avail_end's date, fall back to that date —
       the previous publishing day will have a complete OI/quotes window.
    2. Build defs/quotes/stats windows on the effective date as before.
    3. Clamp every end timestamp to min(query_end, avail_end).
    4. If snapshot_time itself is past avail_end on the effective date,
       slide the quote window back so it ends at avail_end.

    `avail_end` may be None (metadata fetch failed) — in that case we
    return the unclamped windows and let Databento raise as before.
    """
    # Choose the most recent date on which the requested snapshot window
    # fully fits inside the dataset's published range, skipping weekends.
    # This handles three cases:
    #  - today not yet published     → step back to most recent data day
    #  - today partially published   → step back if snapshot_time isn't
    #                                  yet covered (avoids querying the
    #                                  exact 09:30 ET boundary, which has
    #                                  no quotes flowing yet)
    #  - holidays                    → ride on Databento's own avail_end,
    #                                  which already excludes them
    effective = snapshot_date
    if avail_end is not None:
        hour, minute = (int(x) for x in snapshot_time.split(":"))
        most_recent_data_date = (avail_end - timedelta(seconds=1)).date()
        candidate = min(snapshot_date, most_recent_data_date)
        for _ in range(8):
            snap_start = datetime(
                candidate.year, candidate.month, candidate.day,
                hour, minute, tzinfo=timezone.utc,
            )
            snap_end = snap_start + timedelta(seconds=snapshot_window_seconds)
            if snap_end <= avail_end and candidate.weekday() < 5:
                break
            candidate = candidate - timedelta(days=1)
        if candidate != snapshot_date:
            logger.warning(
                f"  {dataset}: requested {snapshot_date} snapshot window doesn't "
                f"fit (avail_end={avail_end.isoformat()}); falling back to {candidate}"
            )
        effective = candidate

    date_str = effective.isoformat()
    next_date_str = (effective + timedelta(days=1)).isoformat()

    def_start = date_str
    def_end = next_date_str

    quote_start_dt = datetime.strptime(f"{date_str}T{snapshot_time}", "%Y-%m-%dT%H:%M")
    quote_end_dt = quote_start_dt + timedelta(seconds=snapshot_window_seconds)
    quote_start = quote_start_dt.isoformat()
    quote_end = quote_end_dt.isoformat()

    if dataset == "OPRA.PILLAR":
        stats_start = f"{date_str}T09:30"
        stats_end = f"{date_str}T12:00"
    elif dataset == "GLBX.MDP3":
        stats_start = f"{date_str}T00:00"
        stats_end = f"{date_str}T02:00"
    else:
        stats_start = date_str
        stats_end = next_date_str

    if avail_end is not None:
        avail_end_ts = pd.Timestamp(avail_end).tz_convert("UTC")
        avail_end_iso = avail_end_ts.isoformat()

        def clamp_end(end_str_local: str) -> str:
            return avail_end_iso if _to_utc(end_str_local) > avail_end_ts else end_str_local

        def_end = clamp_end(def_end)
        stats_end = clamp_end(stats_end)
        quote_end = clamp_end(quote_end)

        # If the quote window's start itself is past avail_end (e.g. tick
        # fired at 16:00 ET but the feed only has 15:50 ET), slide back.
        if _to_utc(quote_start) > avail_end_ts:
            slide_end = avail_end_ts
            slide_start = slide_end - pd.Timedelta(seconds=snapshot_window_seconds)
            logger.warning(
                f"  {dataset}: snapshot_time {snapshot_time} on {date_str} is past "
                f"available end {avail_end_iso}; sliding quote window back."
            )
            quote_start = slide_start.isoformat()
            quote_end = slide_end.isoformat()

    return {
        "effective_date": effective,
        "def_start": def_start,
        "def_end": def_end,
        "quote_start": quote_start,
        "quote_end": quote_end,
        "stats_start": stats_start,
        "stats_end": stats_end,
    }


def ingest_chain(
    product: ProductConfig,
    snapshot_date: date,
    snapshot_time: str = "15:55",
    snapshot_window_seconds: int = 1,
) -> dict:
    """
    Pull raw option chain data from Databento.

    Parameters
    ----------
    product : ProductConfig
    snapshot_date : date
    snapshot_time : str - HH:MM for quote snapshot start (UTC naive, as
        Databento expects)
    snapshot_window_seconds : int - quote window length in seconds. Pulls the
        consolidated top-of-book snapshot schema (cmbp-1 for OPRA, mbp-1 for
        GLBX) over a tight window and takes the last quote per instrument.

    The query window is clamped to the dataset's actually-available range
    via metadata.get_dataset_range; if the requested date is past the
    feed's high-water mark we fall back to the most recent published day
    so the scheduler still produces output during the T+0 publish lag.

    Returns
    -------
    dict with keys: definitions, quotes, stats (all DataFrames),
    plus effective_date (the date actually queried after fallback).
    """
    client = _build_client()
    ds = product.dataset
    sym = product.parent_symbol

    avail_end = _get_dataset_available_end(client, ds)
    window = resolve_query_window(
        avail_end, ds, snapshot_date, snapshot_time, snapshot_window_seconds
    )
    effective_date = window["effective_date"]
    date_str = effective_date.isoformat()

    logger.info(f"Ingesting {product.symbol} chain for {date_str} from {ds}")
    if avail_end is not None:
        logger.info(f"  Dataset {ds} available_end={avail_end.isoformat()}")
    if effective_date != snapshot_date:
        logger.info(f"  Effective date adjusted from {snapshot_date} → {effective_date}")

    # 1. Instrument definitions
    logger.info(f"  Pulling definitions ({window['def_start']} → {window['def_end']})...")
    defs_data = client.timeseries.get_range(
        dataset=ds,
        schema="definition",
        symbols=[sym],
        stype_in="parent",
        start=window["def_start"],
        end=window["def_end"],
    )
    defs_df = defs_data.to_df()

    # 2. Top-of-book quote snapshot.
    # Use the consolidated MBP-1 schema (cmbp-1 on OPRA, mbp-1 on GLBX) over
    # a tight window — typically 1 second — at the snapshot time. We take the
    # last update per instrument from within the window in build_chain_dataframe.
    # This pulls dramatically less data than the cbbo-1m bar schema and avoids
    # the gateway timeouts that came with multi-minute aggregated bar pulls.
    quotes_schema = "cmbp-1" if ds == "OPRA.PILLAR" else "mbp-1"
    logger.info(
        f"  Pulling quote snapshot via {quotes_schema} "
        f"({window['quote_start']} → {window['quote_end']})..."
    )
    quotes_data = client.timeseries.get_range(
        dataset=ds,
        schema=quotes_schema,
        symbols=[sym],
        stype_in="parent",
        start=window["quote_start"],
        end=window["quote_end"],
    )
    quotes_df = quotes_data.to_df()

    # 3. Daily statistics (open interest)
    # OI lands in a short burst once per day; pulling the full 24h is millions
    # of non-OI rows. Use a narrow per-dataset window around the known publish time.
    logger.info(
        f"  Pulling statistics (OI) from {window['stats_start']} to {window['stats_end']}..."
    )
    stats_data = client.timeseries.get_range(
        dataset=ds,
        schema="statistics",
        symbols=[sym],
        stype_in="parent",
        start=window["stats_start"],
        end=window["stats_end"],
    )
    stats_df = stats_data.to_df()

    return {
        "definitions": defs_df,
        "quotes": quotes_df,
        "stats": stats_df,
        "effective_date": effective_date,
    }


def build_chain_dataframe(raw: dict, product: ProductConfig) -> pd.DataFrame:
    """
    Join definitions, quotes, and stats into a single DataFrame
    with one row per contract.

    Columns: instrument_id, strike, expiry, is_call, bid, ask, mid_price, oi
    """
    defs = raw["definitions"]
    quotes = raw["quotes"]
    stats = raw["stats"]

    # ── Parse definitions ──
    # Extract strike, expiry date, option type from definitions
    # Column names depend on Databento schema version; adapt as needed
    def_cols = defs.columns.tolist()
    logger.info(f"  Definition columns: {def_cols[:15]}...")

    # Databento definition schema provides:
    #   instrument_id, strike_price, expiration, instrument_class (C/P)
    # Prices returned by to_df() are already decimal (SDK unscales 1e-9 fixed-point).
    def_parsed = defs[["instrument_id", "strike_price", "expiration", "instrument_class"]].copy()
    def_parsed = def_parsed.drop_duplicates(subset=["instrument_id"])
    def_parsed["strike"] = def_parsed["strike_price"].astype(float)
    def_parsed["expiry"] = pd.to_datetime(def_parsed["expiration"]).dt.tz_localize(None).dt.normalize()
    def_parsed["is_call"] = def_parsed["instrument_class"].astype(str).isin(["C", "CE", "CALL"])
    def_parsed = def_parsed[["instrument_id", "strike", "expiry", "is_call"]]

    # ── Parse quotes ──
    # Take last quote per instrument for the snapshot window
    quotes_last = quotes.sort_values("ts_event").groupby("instrument_id").last()
    quotes_last = quotes_last[["bid_px_00", "ask_px_00"]].reset_index()
    quotes_last.columns = ["instrument_id", "bid", "ask"]
    quotes_last["bid"] = quotes_last["bid"].astype(float)
    quotes_last["ask"] = quotes_last["ask"].astype(float)
    quotes_last["mid_price"] = (quotes_last["bid"] + quotes_last["ask"]) / 2

    # ── Parse stats ──
    # stat_type == 9  is OPEN_INTEREST and stat_type == 6 is CLEARED_VOLUME
    # on both OPRA.PILLAR and GLBX.MDP3.
    # Quantity of 2147483647 (INT32_MAX) is the unset sentinel — drop those.
    if "stat_type" in stats.columns:
        avail_types = sorted(stats["stat_type"].dropna().unique().tolist())
        logger.info(f"  Stats stat_types available: {avail_types}")

    oi_stats = stats[stats["stat_type"] == 9].copy()
    if len(oi_stats) == 0:
        logger.warning("  No OI rows (stat_type=9) found; OI will default to 0.")
    else:
        oi_stats = oi_stats[oi_stats["quantity"] < 2_147_483_647]

    oi_last = oi_stats.sort_values("ts_event").groupby("instrument_id").last()
    oi_last = oi_last[["quantity"]].reset_index()
    oi_last.columns = ["instrument_id", "oi"]
    oi_last["oi"] = oi_last["oi"].astype(float)

    # Volume: stat_type == 6 is CLEARED_VOLUME (daily traded volume per contract)
    vol_stats = stats[stats["stat_type"] == 6].copy()
    if len(vol_stats) == 0:
        logger.info("  No volume rows (stat_type=6) found; volume will default to 0.")
        vol_last = pd.DataFrame(columns=["instrument_id", "volume"])
    else:
        vol_stats = vol_stats[vol_stats["quantity"] < 2_147_483_647]
        vol_last = vol_stats.sort_values("ts_event").groupby("instrument_id").last()
        vol_last = vol_last[["quantity"]].reset_index()
        vol_last.columns = ["instrument_id", "volume"]
        vol_last["volume"] = vol_last["volume"].astype(float)

    # ── Join ──
    chain = def_parsed.merge(quotes_last, on="instrument_id", how="inner")
    chain = chain.merge(oi_last, on="instrument_id", how="left")
    chain = chain.merge(vol_last, on="instrument_id", how="left")
    chain["oi"] = chain["oi"].fillna(0).astype(int)
    chain["volume"] = chain["volume"].fillna(0).astype(int)
    chain["volume_to_oi_ratio"] = chain["volume"] / chain["oi"].clip(lower=1)

    logger.info(f"  Chain built: {len(chain)} contracts")
    return chain


# ══════════════════════════════════════════════════════════════
# FILTERING
# ══════════════════════════════════════════════════════════════

def filter_chain(
    df: pd.DataFrame,
    underlying_price: float,
    snapshot_date: date,
    filters: FilterConfig = FILTER_DEFAULTS,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply data quality filters.

    Returns
    -------
    filtered : DataFrame
    filter_log : dict with counts of contracts removed by each filter
    """
    n_start = len(df)
    log = {"start": n_start}

    # Compute DTE
    df = df.copy()
    df["dte"] = (df["expiry"] - pd.Timestamp(snapshot_date)).dt.days

    # 1. Minimum OI
    mask_oi = df["oi"] >= filters.min_open_interest
    log["removed_low_oi"] = int((~mask_oi).sum())

    # 2. Valid bid
    mask_bid = df["bid"] > filters.min_bid
    log["removed_no_bid"] = int((~mask_bid).sum())

    # 3. Max spread ratio
    spread_ratio = (df["ask"] - df["bid"]) / df["mid_price"]
    mask_spread = spread_ratio <= filters.max_spread_ratio
    # Handle NaN/inf from zero mid
    mask_spread = mask_spread.fillna(False)
    log["removed_wide_spread"] = int((~mask_spread).sum())

    # 4. Moneyness window
    lower = underlying_price * (1 - filters.moneyness_range)
    upper = underlying_price * (1 + filters.moneyness_range)
    mask_money = (df["strike"] >= lower) & (df["strike"] <= upper)
    log["removed_moneyness"] = int((~mask_money).sum())

    # 5. Minimum DTE
    mask_dte = df["dte"] >= filters.min_dte_days
    log["removed_low_dte"] = int((~mask_dte).sum())

    # Apply all filters
    mask_all = mask_oi & mask_bid & mask_spread & mask_money & mask_dte
    filtered = df[mask_all].copy()

    log["after_filter"] = len(filtered)
    log["total_removed"] = n_start - len(filtered)
    logger.info(f"  Filtered: {n_start} → {len(filtered)} contracts ({log['total_removed']} removed)")

    return filtered, log


# ══════════════════════════════════════════════════════════════
# IV SOLVE + GREEKS
# ══════════════════════════════════════════════════════════════

def solve_and_compute(
    df: pd.DataFrame,
    underlying_price: float,
    product: ProductConfig,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Run IV solver then compute Greeks for all contracts.

    Returns
    -------
    df : DataFrame with iv, delta, gamma, vanna, charm columns added
    iv_log : dict with convergence stats
    """
    S = underlying_price
    K = df["strike"].values.astype(float)
    T = df["dte"].values.astype(float) / 365.0
    is_call = df["is_call"].values
    market_price = df["mid_price"].values.astype(float)
    r = risk_free_rate
    q = product.dividend_yield if product.dividend_yield is not None else 0.0

    logger.info(f"  Solving IV for {len(df)} contracts...")
    t0 = time.time()

    iv, converged = solve_iv(
        market_price=market_price,
        S=S, K=K, T=T, r=r,
        is_call=is_call,
        model=product.pricing_model,
        q=q,
    )

    iv_time = time.time() - t0
    n_converged = int(converged.sum())
    n_failed = int((~converged).sum())
    logger.info(f"  IV solved: {n_converged} converged, {n_failed} failed ({iv_time:.2f}s)")

    iv_log = {
        "total": len(df),
        "converged": n_converged,
        "failed": n_failed,
        "solve_time_seconds": round(iv_time, 3),
    }

    # Add IV to dataframe and filter out failures
    df = df.copy()
    df["iv"] = iv
    df = df[converged].copy()

    # Recompute T for the filtered set
    T_filtered = df["dte"].values.astype(float) / 365.0
    K_filtered = df["strike"].values.astype(float)
    iv_filtered = df["iv"].values
    is_call_filtered = df["is_call"].values

    logger.info(f"  Computing Greeks for {len(df)} contracts...")
    greeks = compute_greeks(
        S=S,
        K=K_filtered,
        T=T_filtered,
        r=r,
        sigma=iv_filtered,
        is_call=is_call_filtered,
        model=product.pricing_model,
        q=q,
    )

    df["delta"] = greeks["delta"]
    df["gamma"] = greeks["gamma"]
    df["vanna"] = greeks["vanna"]
    df["charm"] = greeks["charm"]

    return df, iv_log


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(
    symbol: str,
    snapshot_date: date,
    underlying_price: Optional[float] = None,
    risk_free_rate: float = 0.05,
    snapshot_time: str = "15:55",
    filters: FilterConfig = FILTER_DEFAULTS,
) -> dict:
    """
    Execute the full computation pipeline for a single product.

    Steps:
      1. Load product config
      2. Ingest chain from Databento
      3. Build chain DataFrame
      4. Filter
      5. IV solve + Greek computation
      6. Aggregate (GEX, VEX, CEX, profiles, breakdowns)

    Parameters
    ----------
    symbol : str - Product symbol (e.g., "SPY")
    snapshot_date : date
    underlying_price : float - Current spot or futures price
    risk_free_rate : float
    snapshot_time : str - HH:MM for quote snapshot
    filters : FilterConfig

    Returns
    -------
    dict with keys:
      - product: ProductConfig
      - snapshot_date: date
      - underlying_price: float
      - risk_free_rate: float
      - scores: dict (gex, vex, cex, gex_plus)
      - strike_profiles: DataFrame
      - expiry_breakdown: DataFrame
      - contracts: DataFrame (full per-contract detail)
      - metadata: dict (filter_log, iv_log, timing)
    """
    t_start = time.time()

    # 1. Product config
    if symbol not in PRODUCTS:
        raise ValueError(f"Unknown product: {symbol}. Available: {list(PRODUCTS.keys())}")
    product = PRODUCTS[symbol]
    logger.info(f"Pipeline start: {symbol} on {snapshot_date}")

    # 2. Ingest
    raw = ingest_chain(product, snapshot_date, snapshot_time)

    # If the dataset's published range didn't include the requested date,
    # ingest_chain falls back to the most recent published day. Use that
    # date downstream so DTE, parity inference, and archive metadata all
    # reflect the data actually queried.
    effective_date = raw.get("effective_date", snapshot_date)
    if effective_date != snapshot_date:
        logger.info(f"  run_pipeline using effective snapshot_date={effective_date}")
        snapshot_date = effective_date

    # 3. Build chain
    chain = build_chain_dataframe(raw, product)

    # 3b. Infer underlying price if not provided
    if underlying_price is None:
        underlying_price = infer_underlying_price(
            chain, snapshot_date, product, risk_free_rate
        )
        logger.info(f"  Inferred underlying for {symbol}: {underlying_price:.3f}")

    # 4. Filter
    filtered, filter_log = filter_chain(chain, underlying_price, snapshot_date, filters)

    if len(filtered) == 0:
        logger.error("No contracts remaining after filtering!")
        return {
            "product": product,
            "snapshot_date": snapshot_date,
            "underlying_price": underlying_price,
            "risk_free_rate": risk_free_rate,
            "scores": {"gex": 0, "vex": 0, "cex": 0, "gex_plus": 0},
            "strike_profiles": pd.DataFrame(),
            "expiry_breakdown": pd.DataFrame(),
            "contracts": pd.DataFrame(),
            "metadata": {"filter_log": filter_log, "iv_log": {}, "total_time": 0},
        }

    # 5. IV solve + Greeks
    greeks_df, iv_log = solve_and_compute(
        filtered, underlying_price, product, risk_free_rate
    )

    # 6. Aggregate
    outputs = build_all_outputs(greeks_df, underlying_price, product)

    t_total = time.time() - t_start
    logger.info(f"Pipeline complete: {symbol} in {t_total:.2f}s")

    # Build metadata
    metadata = {
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        "product": symbol,
        "underlying_price": underlying_price,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": product.dividend_yield,
        "vol_spot_multiplier": product.vol_spot_multiplier,
        "filter_log": filter_log,
        "iv_log": iv_log,
        "total_time_seconds": round(t_total, 3),
        "engine_version": "1.2",
    }

    return {
        "product": product,
        "snapshot_date": snapshot_date,
        "underlying_price": underlying_price,
        "risk_free_rate": risk_free_rate,
        "scores": outputs["scores"],
        "strike_profiles": outputs["strike_profiles"],
        "expiry_breakdown": outputs["expiry_breakdown"],
        "contracts": outputs["contracts"],
        "metadata": metadata,
    }


# ══════════════════════════════════════════════════════════════
# MULTI-PRODUCT RUN
# ══════════════════════════════════════════════════════════════

def run_all(
    snapshot_date: Optional[date] = None,
    risk_free_rate: float = 0.05,
    snapshot_time: str = "15:55",
    archive: bool = True,
) -> dict:
    """
    Run the full pipeline for every configured product in sequence.

    Each product's underlying price is auto-inferred via put-call parity.
    Failures on one product do not abort the run; the error is logged and the
    next product is attempted.

    Returns
    -------
    dict with:
      - results: per-symbol dict of pipeline output (or None on failure)
      - errors: per-symbol error string (or None on success)
      - summary: list of {symbol, ok, gex, vex, cex, gex_plus, underlying}
    """
    from archive import archive_results

    if snapshot_date is None:
        snapshot_date = date.today()

    results = {}
    errors = {}
    summary = []

    for symbol in PRODUCTS.keys():
        t0 = time.time()
        try:
            logger.info(f"\n=== run_all: {symbol} ===")
            res = run_pipeline(
                symbol=symbol,
                snapshot_date=snapshot_date,
                underlying_price=None,
                risk_free_rate=risk_free_rate,
                snapshot_time=snapshot_time,
            )
            results[symbol] = res
            errors[symbol] = None

            if archive:
                try:
                    archive_results(res)
                except Exception as ae:
                    logger.error(f"  archive_results failed for {symbol}: {ae}")

            summary.append({
                "symbol": symbol,
                "ok": True,
                "underlying": res["underlying_price"],
                "gex": res["scores"]["gex"],
                "vex": res["scores"]["vex"],
                "cex": res["scores"]["cex"],
                "gex_plus": res["scores"]["gex_plus"],
                "duration_s": round(time.time() - t0, 2),
            })
        except Exception as e:
            logger.error(f"  {symbol} failed: {type(e).__name__}: {e}")
            results[symbol] = None
            errors[symbol] = f"{type(e).__name__}: {e}"
            summary.append({
                "symbol": symbol,
                "ok": False,
                "error": errors[symbol],
                "duration_s": round(time.time() - t0, 2),
            })

    return {"results": results, "errors": errors, "summary": summary}


# ══════════════════════════════════════════════════════════════
# PIPELINE WITH SYNTHETIC DATA (for testing without Databento)
# ══════════════════════════════════════════════════════════════

def run_pipeline_synthetic(
    symbol: str = "SPY",
    underlying_price: float = 550.0,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Run the pipeline with synthetic option chain data.
    Useful for testing without Databento API access.
    """
    product = PRODUCTS[symbol]
    snapshot_date = date.today()
    S = underlying_price
    q = product.dividend_yield or 0.0

    # Generate synthetic chain
    np.random.seed(42)
    strikes = np.arange(S * 0.85, S * 1.15, 1.0)
    expiries_dte = [2, 5, 14, 30, 60]
    rows = []

    for K in strikes:
        for dte in expiries_dte:
            for is_call in [True, False]:
                T = dte / 365.0
                moneyness = np.log(S / K) / np.sqrt(T)
                # Synthetic IV: smile shape
                base_iv = 0.18 + 0.05 * (K / S - 1.0)**2
                iv = base_iv + np.random.normal(0, 0.005)
                iv = max(iv, 0.05)

                # Compute BSM price for this contract
                from models import bsm_call_price, bsm_put_price
                if is_call:
                    price = bsm_call_price(S, K, T, risk_free_rate, q, iv)
                else:
                    price = bsm_put_price(S, K, T, risk_free_rate, q, iv)

                if isinstance(price, np.ndarray):
                    price = float(price[0]) if len(price) > 0 else float(price)

                # Synthetic OI: higher near ATM
                oi = max(10, int(5000 * np.exp(-0.5 * ((K - S) / (S * 0.05))**2)))
                oi += np.random.randint(0, 500)

                volume = max(0, int(oi * np.random.uniform(0.05, 0.3)))
                rows.append({
                    "instrument_id": len(rows),
                    "strike": K,
                    "expiry": pd.Timestamp(snapshot_date) + pd.Timedelta(days=dte),
                    "is_call": is_call,
                    "bid": max(price * 0.95, 0.01),
                    "ask": price * 1.05,
                    "mid_price": price,
                    "oi": oi,
                    "volume": volume,
                    "volume_to_oi_ratio": volume / max(oi, 1),
                    "dte": dte,
                })

    chain = pd.DataFrame(rows)

    # Filter
    filtered, filter_log = filter_chain(chain, S, snapshot_date)

    # IV solve + Greeks
    greeks_df, iv_log = solve_and_compute(filtered, S, product, risk_free_rate)

    # Aggregate
    outputs = build_all_outputs(greeks_df, S, product)

    metadata = {
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        "product": symbol,
        "underlying_price": S,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": q,
        "vol_spot_multiplier": product.vol_spot_multiplier,
        "filter_log": filter_log,
        "iv_log": iv_log,
        "total_time_seconds": 0,
        "engine_version": "1.2",
        "data_source": "synthetic",
    }

    return {
        "product": product,
        "snapshot_date": snapshot_date,
        "underlying_price": S,
        "risk_free_rate": risk_free_rate,
        "scores": outputs["scores"],
        "strike_profiles": outputs["strike_profiles"],
        "expiry_breakdown": outputs["expiry_breakdown"],
        "contracts": outputs["contracts"],
        "metadata": metadata,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "=" * 60)
    print("SYNTHETIC PIPELINE TEST")
    print("=" * 60)

    result = run_pipeline_synthetic("SPY", 550.0, 0.05)
    scores = result["scores"]

    print(f"\n  Underlying: SPY @ ${result['underlying_price']:.2f}")
    print(f"  Risk-free rate: {result['risk_free_rate']:.2%}")
    print(f"  Contracts computed: {len(result['contracts'])}")
    print(f"\n  ── Headline Scores ──")
    print(f"  GEX:   ${scores['gex']:>15,.0f} /pt")
    print(f"  VEX:   ${scores['vex']:>15,.0f} /pt")
    print(f"  CEX:   ${scores['cex']:>15,.0f} /day")
    print(f"  GEX+:  ${scores['gex_plus']:>15,.0f} /pt")
    print(f"\n  Strike profiles: {len(result['strike_profiles'])} strikes")
    print(f"  Expiry breakdown: {len(result['expiry_breakdown'])} strike/bucket combos")
    print(f"\n  Filter log: {result['metadata']['filter_log']}")
    print(f"  IV log: {result['metadata']['iv_log']}")
    print("=" * 60)
