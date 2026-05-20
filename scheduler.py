#!/usr/bin/env python3
"""
Greek Engine Scheduler — 30-minute auto-refresh during US market hours.

Runs run_all() at :00 and :30 of every hour from 09:00 through 17:00
Eastern, Monday–Friday, skipping US market holidays. Outside that window,
the loop sleeps until the next valid tick.

Usage:
    python scheduler.py          # persistent service mode
    python scheduler.py --once   # single run-now for manual testing
"""

import argparse
import logging
import sys
import time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

from pipeline import run_all


ET = ZoneInfo("America/New_York")

MARKET_START_HOUR = 9   # 09:00 ET inclusive
MARKET_END_HOUR = 17    # 17:00 ET inclusive (last tick at 17:00)
TICK_MINUTES = 30


# US market holidays — 2026 baseline. Extend yearly.
US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed; Jul 4 is Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

US_MARKET_HOLIDAYS = US_MARKET_HOLIDAYS_2026

logger = logging.getLogger("scheduler")


def is_market_hours(now_et: datetime) -> bool:
    """Weekday, not a holiday, and within 09:00–17:00 ET."""
    if now_et.weekday() >= 5:
        return False
    if now_et.date() in US_MARKET_HOLIDAYS:
        return False
    if not (MARKET_START_HOUR <= now_et.hour <= MARKET_END_HOUR):
        return False
    # Last tick is at 17:00 sharp.
    if now_et.hour == MARKET_END_HOUR and now_et.minute > 0:
        return False
    return True


def next_tick(now_et: datetime) -> datetime:
    """Return the next :00 / :30 boundary inside market hours."""
    candidate = now_et.replace(second=0, microsecond=0)
    if candidate.minute < 30:
        candidate = candidate.replace(minute=30)
    else:
        candidate = (candidate + timedelta(hours=1)).replace(minute=0)

    # Walk forward until the candidate is a valid market-hours tick.
    while True:
        if is_market_hours(candidate):
            return candidate
        # Jump cheaply: if before 09:00 on a weekday, snap to 09:00.
        if (
            candidate.weekday() < 5
            and candidate.date() not in US_MARKET_HOLIDAYS
            and candidate.hour < MARKET_START_HOUR
        ):
            return candidate.replace(hour=MARKET_START_HOUR, minute=0)
        # If after market end (or any non-market day), advance to next day 09:00.
        candidate = (candidate + timedelta(days=1)).replace(
            hour=MARKET_START_HOUR, minute=0
        )


def run_tick(rate: float) -> None:
    """Execute one multi-product run; never raise."""
    t0 = time.time()
    try:
        out = run_all(risk_free_rate=rate)
        ok_count = sum(1 for r in out["summary"] if r.get("ok"))
        fail_count = len(out["summary"]) - ok_count
        logger.info(
            f"Tick complete: {ok_count} ok, {fail_count} failed, "
            f"{time.time() - t0:.1f}s"
        )
        for row in out["summary"]:
            if row["ok"]:
                logger.info(
                    f"  {row['symbol']}: spot={row['underlying']:.2f}  "
                    f"GEX={row['gex']:,.0f}  VEX={row['vex']:,.0f}  "
                    f"CEX={row['cex']:,.0f}"
                )
            else:
                logger.error(f"  {row['symbol']}: {row.get('error')}")
    except Exception as e:
        logger.exception(f"Tick raised: {type(e).__name__}: {e}")


def loop_forever(rate: float) -> None:
    logger.info("Scheduler started (30-minute ticks during 09:00–17:00 ET, Mon–Fri).")
    while True:
        now = datetime.now(ET)
        tick = next_tick(now)
        sleep_s = max(0.0, (tick - now).total_seconds())
        logger.info(
            f"Sleeping {sleep_s/60:.1f} min until next tick at {tick.isoformat()}"
        )
        time.sleep(sleep_s + 1)  # +1s so the moment-of clock comparison passes
        now = datetime.now(ET)
        if is_market_hours(now):
            logger.info(f"Tick at {now.isoformat()} (rate={rate})")
            run_tick(rate)
        else:
            # Skipped — likely a holiday flagged after sleep started.
            logger.info(f"Skipping tick at {now.isoformat()}: outside market hours")


def main():
    parser = argparse.ArgumentParser(description="Greek Engine 30-minute scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run a single multi-product tick now and exit.")
    parser.add_argument("--rate", type=float, default=0.05,
                        help="Risk-free rate (default: 0.05)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    if args.once:
        logger.info("Manual --once tick.")
        run_tick(args.rate)
        return 0

    try:
        loop_forever(args.rate)
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
