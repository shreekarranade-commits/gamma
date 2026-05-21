"""
Positioning aggregation: raw OI + Volume sums per (product, expiry).

This module bypasses the Greeks engine entirely. It produces a simple
calls/puts/net positioning concentration signal from open interest and
traded volume across the full option chain — no IV solve, no BSM,
no per-contract delta or gamma.

Pure-functional: no I/O, no globals.
"""

import logging
from datetime import date, datetime, time, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)


ET = ZoneInfo("America/New_York")


class PositioningResult(NamedTuple):
    calls_total: float
    puts_total: float
    net: float  # calls_total - puts_total


# ══════════════════════════════════════════════════════════════
# CORE COMPUTE
# ══════════════════════════════════════════════════════════════

def _is_call_mask(chain: pd.DataFrame) -> pd.Series:
    """
    Accept either a 'type' column ('C' / 'P') or an 'is_call' boolean.
    Returns a boolean Series aligned with the input index.
    """
    if "type" in chain.columns:
        return chain["type"].astype(str).str.upper().str.startswith("C")
    if "is_call" in chain.columns:
        return chain["is_call"].astype(bool)
    raise ValueError("chain must have a 'type' or 'is_call' column")


def compute_positioning(chain: pd.DataFrame, expiry: date) -> PositioningResult:
    """
    Compute calls total, puts total, and net OI+Volume for a single expiry.

    chain must have columns: strike, type ('C' or 'P') or is_call (bool),
    expiry, oi, volume.

    Strikes with zero OI AND zero volume are excluded — they contribute
    zero anyway, but excluding them keeps the count honest for diagnostics.

    Returns
    -------
    PositioningResult(calls_total, puts_total, net) summed across all
    non-zero strikes in the given expiry. If the expiry has no contracts
    at all, returns PositioningResult(0.0, 0.0, 0.0) and logs a warning.
    """
    if chain is None or len(chain) == 0:
        logger.warning(f"compute_positioning: empty chain for expiry {expiry}")
        return PositioningResult(0.0, 0.0, 0.0)

    target = pd.Timestamp(expiry).normalize()
    expiry_norm = pd.to_datetime(chain["expiry"]).dt.tz_localize(None).dt.normalize()
    rows = chain[expiry_norm == target]

    if rows.empty:
        logger.warning(f"compute_positioning: no contracts for expiry {expiry}")
        return PositioningResult(0.0, 0.0, 0.0)

    oi = rows["oi"].fillna(0).astype(float)
    vol = rows["volume"].fillna(0).astype(float)

    nonzero_mask = (oi > 0) | (vol > 0)
    rows = rows[nonzero_mask]
    oi = oi[nonzero_mask]
    vol = vol[nonzero_mask]

    if rows.empty:
        return PositioningResult(0.0, 0.0, 0.0)

    is_call = _is_call_mask(rows)
    calls_total = float((oi[is_call] + vol[is_call]).sum())
    puts_total = float((oi[~is_call] + vol[~is_call]).sum())
    net = calls_total - puts_total

    return PositioningResult(calls_total, puts_total, net)


def compute_positioning_all_expiries(chain: pd.DataFrame) -> dict[date, PositioningResult]:
    """
    Returns {expiry_date: PositioningResult} for every unique expiry in the chain.
    """
    if chain is None or len(chain) == 0 or "expiry" not in chain.columns:
        return {}

    expiry_norm = pd.to_datetime(chain["expiry"]).dt.tz_localize(None).dt.normalize()
    unique = sorted({d.date() for d in expiry_norm.unique() if not pd.isna(d)})

    return {e: compute_positioning(chain, e) for e in unique}


# ══════════════════════════════════════════════════════════════
# SNAPSHOT TIMING
# ══════════════════════════════════════════════════════════════

def current_snapshot_time_et(now: datetime | None = None) -> time:
    """
    Return the wall-clock ET time of the current snapshot, with seconds
    and microseconds zeroed. Used as the snapshot_time column value
    when archiving from the scheduler or manual CLI.
    """
    if now is None:
        now = datetime.now(ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    return time(hour=now.hour, minute=now.minute, second=0)


# ══════════════════════════════════════════════════════════════
# DASHBOARD HELPERS (pure functions, tested directly)
# ══════════════════════════════════════════════════════════════

def current_week_dates(dates: list[date]) -> set[date]:
    """
    Return the Mon–Fri block (set of 5 dates) containing the most recent
    date in `dates`. Empty set if no dates are given.
    """
    if not dates:
        return set()
    latest = max(dates)
    monday = latest - timedelta(days=latest.weekday())
    return {monday + timedelta(days=i) for i in range(5)}


def sum_positioning_across_expiries(
    df: pd.DataFrame, selected: list[date]
) -> pd.DataFrame:
    """
    Sum calls_total and puts_total across the chosen expiries per snapshot_time.

    Returns DataFrame with columns: snapshot_time, calls, puts.
    Empty DataFrame if `df` is empty, `selected` is empty, or no rows match.
    """
    cols = ["snapshot_time", "calls", "puts"]
    if df is None or df.empty or not selected:
        return pd.DataFrame(columns=cols)

    selected_dates = {pd.Timestamp(d).date() if not isinstance(d, date) else d
                       for d in selected}
    expiry_dates = pd.to_datetime(df["expiry"]).dt.date
    sub = df[expiry_dates.isin(selected_dates)]
    if sub.empty:
        return pd.DataFrame(columns=cols)

    out = sub.groupby("snapshot_time", as_index=False).agg(
        calls=("calls_total", "sum"),
        puts=("puts_total", "sum"),
    )
    return out.sort_values("snapshot_time").reset_index(drop=True)
