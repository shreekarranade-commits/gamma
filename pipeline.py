"""
Data pipeline: ingest → join → filter → IV solve → Greeks → aggregate.

Handles both OPRA.PILLAR (equities) and GLBX.MDP3 (futures) datasets.
"""

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


def ingest_chain(
    product: ProductConfig,
    snapshot_date: date,
    snapshot_time: str = "15:45",
    snapshot_window_minutes: int = 15,
) -> dict:
    """
    Pull raw option chain data from Databento.

    Parameters
    ----------
    product : ProductConfig
    snapshot_date : date
    snapshot_time : str - HH:MM for quote snapshot start (UTC naive, as Databento expects)
    snapshot_window_minutes : int - quote window length; we take the last quote
        per instrument from within this window

    Returns
    -------
    dict with keys: definitions, quotes, stats (all DataFrames)
    """
    client = _build_client()
    ds = product.dataset
    sym = product.parent_symbol
    date_str = snapshot_date.isoformat()
    next_date = (snapshot_date + timedelta(days=1)).isoformat()

    logger.info(f"Ingesting {product.symbol} chain for {date_str} from {ds}")

    # 1. Instrument definitions
    logger.info("  Pulling definitions...")
    defs_data = client.timeseries.get_range(
        dataset=ds,
        schema="definition",
        symbols=[sym],
        stype_in="parent",
        start=date_str,
        end=next_date,
    )
    defs_df = defs_data.to_df()

    # 2. Top-of-book quotes (bounded window)
    start_dt = datetime.strptime(f"{date_str}T{snapshot_time}", "%Y-%m-%dT%H:%M")
    end_dt = start_dt + timedelta(minutes=snapshot_window_minutes)
    start_ts = start_dt.isoformat()
    end_ts = end_dt.isoformat()
    logger.info(f"  Pulling quotes ({start_ts} to {end_ts})...")
    # 1-minute aggregated BBO: keeps the stream well under the 5 GB limit
    # and is the right granularity for an EOD snapshot.
    quotes_schema = "cbbo-1m" if ds == "OPRA.PILLAR" else "bbo-1m"
    quotes_data = client.timeseries.get_range(
        dataset=ds,
        schema=quotes_schema,
        symbols=[sym],
        stype_in="parent",
        start=start_ts,
        end=end_ts,
    )
    quotes_df = quotes_data.to_df()

    # 3. Daily statistics (open interest)
    # OI lands in a short burst once per day; pulling the full 24h is millions
    # of non-OI rows. Use a narrow per-dataset window around the known publish time.
    if ds == "OPRA.PILLAR":
        stats_start = f"{date_str}T09:30"
        stats_end = f"{date_str}T12:00"
    elif ds == "GLBX.MDP3":
        stats_start = f"{date_str}T00:00"
        stats_end = f"{date_str}T02:00"
    else:
        stats_start = date_str
        stats_end = next_date
    logger.info(f"  Pulling statistics (OI) from {stats_start} to {stats_end}...")
    stats_data = client.timeseries.get_range(
        dataset=ds,
        schema="statistics",
        symbols=[sym],
        stype_in="parent",
        start=stats_start,
        end=stats_end,
    )
    stats_df = stats_data.to_df()

    return {
        "definitions": defs_df,
        "quotes": quotes_df,
        "stats": stats_df,
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
    # stat_type == 9 is OPEN_INTEREST on OPRA.PILLAR / GLBX.MDP3.
    # Quantity of 2147483647 (INT32_MAX) is the unset sentinel — drop those.
    oi_stats = stats[stats["stat_type"] == 9].copy()
    if len(oi_stats) == 0:
        logger.warning("  No OI rows (stat_type=9) found; OI will default to 0.")
    else:
        oi_stats = oi_stats[oi_stats["quantity"] < 2_147_483_647]

    oi_last = oi_stats.sort_values("ts_event").groupby("instrument_id").last()
    oi_last = oi_last[["quantity"]].reset_index()
    oi_last.columns = ["instrument_id", "oi"]
    oi_last["oi"] = oi_last["oi"].astype(float)

    # ── Join ──
    chain = def_parsed.merge(quotes_last, on="instrument_id", how="inner")
    chain = chain.merge(oi_last, on="instrument_id", how="left")
    chain["oi"] = chain["oi"].fillna(0).astype(int)

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
    underlying_price: float,
    risk_free_rate: float = 0.05,
    snapshot_time: str = "15:45",
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

    # 3. Build chain
    chain = build_chain_dataframe(raw, product)

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

                rows.append({
                    "instrument_id": len(rows),
                    "strike": K,
                    "expiry": pd.Timestamp(snapshot_date) + pd.Timedelta(days=dte),
                    "is_call": is_call,
                    "bid": max(price * 0.95, 0.01),
                    "ask": price * 1.05,
                    "mid_price": price,
                    "oi": oi,
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
