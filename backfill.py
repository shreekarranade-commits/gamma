#!/usr/bin/env python3
"""
Historical backfill for the Greek Engine.

Iterates trading days (weekdays, skipping US market holidays) between
--start and --end (inclusive), runs the full pipeline for one product on
each day with the underlying price auto-inferred from the chain via
put-call parity, and archives the result.

A delay is inserted between days to be polite to the Databento API.
Failures on a single day are logged and the loop continues.
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta

from config import PRODUCTS
from scheduler import US_MARKET_HOLIDAYS

logger = logging.getLogger("backfill")


def trading_days(start: date, end: date) -> list[date]:
    """Inclusive list of weekdays in [start, end] that are not US holidays."""
    if end < start:
        return []
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in US_MARKET_HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


def fmt_money(v: float) -> str:
    return f"${v:>15,.0f}"


def run_backfill(
    product: str,
    start: date,
    end: date,
    rate: float = 0.05,
    delay: float = 2.0,
    snapshot_time: str = "15:55",
    dry_run: bool = False,
) -> dict:
    """Backfill a date range. Returns summary dict."""
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product: {product}. Known: {list(PRODUCTS.keys())}")

    days = trading_days(start, end)
    logger.info(
        f"Backfill {product}: {len(days)} trading day(s) between "
        f"{start.isoformat()} and {end.isoformat()}"
    )

    if dry_run:
        for d in days:
            print(f"  [dry-run] {d.isoformat()}  ({d.strftime('%a')})")
        print(f"\n  Total: {len(days)} trading day(s) — no API calls made.")
        return {
            "product": product,
            "days_total": len(days),
            "successes": 0,
            "failures": 0,
            "errors": {},
            "dry_run": True,
        }

    # Imports deferred so --dry-run does not require Databento env.
    from pipeline import run_pipeline
    from archive import archive_results

    successes = 0
    failures = 0
    errors: dict = {}
    t_total = time.time()

    print(
        f"\n  {'DATE':<12} {'STATUS':<6} {'SPOT':>9}  "
        f"{'GEX':>16} {'VEX':>16} {'CEX':>14}  {'TIME':>5}"
    )
    print(f"  {'-' * 92}")

    for i, d in enumerate(days):
        t0 = time.time()
        try:
            result = run_pipeline(
                symbol=product,
                snapshot_date=d,
                underlying_price=None,
                risk_free_rate=rate,
                snapshot_time=snapshot_time,
            )
            archive_results(result)
            s = result["scores"]
            elapsed = time.time() - t0
            successes += 1
            print(
                f"  {d.isoformat():<12} {'OK':<6} "
                f"{result['underlying_price']:>9.2f}  "
                f"{fmt_money(s['gex'])} {fmt_money(s['vex'])} "
                f"{fmt_money(s['cex'])}  {elapsed:>4.1f}s"
            )
        except Exception as e:
            elapsed = time.time() - t0
            failures += 1
            err = f"{type(e).__name__}: {e}"
            errors[d.isoformat()] = err
            print(f"  {d.isoformat():<12} {'FAIL':<6} {err}  ({elapsed:.1f}s)")
            logger.exception(f"Backfill failure on {d}")

        # Polite delay between days, except after the final one.
        if delay > 0 and i < len(days) - 1:
            time.sleep(delay)

    total_elapsed = time.time() - t_total
    summary = {
        "product": product,
        "days_total": len(days),
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "total_time_seconds": round(total_elapsed, 1),
        "dry_run": False,
    }

    print(f"\n  {'=' * 60}")
    print(f"  Backfill summary for {product}")
    print(f"    Range:          {start} → {end}")
    print(f"    Trading days:   {len(days)}")
    print(f"    Successes:      {successes}")
    print(f"    Failures:       {failures}")
    if failures:
        print(f"    Failed dates:   {', '.join(sorted(errors.keys()))}")
    print(f"    Total time:     {total_elapsed:.1f}s")
    if days:
        print(f"    Avg time/day:   {total_elapsed / len(days):.1f}s")
    print(f"  {'=' * 60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Greek Engine historical backfill")
    parser.add_argument("--product", default="SPY", choices=list(PRODUCTS.keys()),
                        help="Product symbol (default: SPY)")
    parser.add_argument("--start", required=True,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (inclusive, default: today)")
    parser.add_argument("--rate", type=float, default=0.05,
                        help="Risk-free rate (default: 0.05)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between days (default: 2)")
    parser.add_argument("--time", default="15:55", dest="snapshot_time",
                        help="Snapshot time HH:MM (default: 15:55)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the dates that would be processed; no API calls.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    if end < start:
        parser.error(f"--end ({end}) precedes --start ({start})")

    summary = run_backfill(
        product=args.product,
        start=start,
        end=end,
        rate=args.rate,
        delay=args.delay,
        snapshot_time=args.snapshot_time,
        dry_run=args.dry_run,
    )

    return 0 if summary["failures"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
