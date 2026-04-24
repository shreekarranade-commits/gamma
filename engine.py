#!/usr/bin/env python3
"""
Greek Exposure Engine - CLI Entry Point

Commands:
  run       Run pipeline for a product (live Databento data)
  synthetic Run pipeline with synthetic data (no API key needed)
  purge     Purge archived data
  serve     Start the dashboard web server
  validate  Run reference case validation
"""

import sys
import argparse
import logging
from datetime import date

from config import PRODUCTS, FILTER_DEFAULTS, ARCHIVE_DEFAULTS


def cmd_run(args):
    """Run pipeline with live Databento data."""
    import os
    os.environ.setdefault("DATABENTO_API_KEY", args.api_key or "")

    from pipeline import run_pipeline
    from archive import archive_results

    result = run_pipeline(
        symbol=args.product,
        snapshot_date=date.fromisoformat(args.date) if args.date else date.today(),
        underlying_price=args.underlying,
        risk_free_rate=args.rate,
        snapshot_time=args.time,
    )

    scores = result["scores"]
    print(f"\n{'=' * 50}")
    print(f"  {args.product} @ ${result['underlying_price']:.2f}")
    print(f"  Date: {result['snapshot_date']}")
    print(f"{'=' * 50}")
    print(f"  GEX:   ${scores['gex']:>15,.0f} /pt")
    print(f"  VEX:   ${scores['vex']:>15,.0f} /pt")
    print(f"  CEX:   ${scores['cex']:>15,.0f} /day")
    print(f"  GEX+:  ${scores['gex_plus']:>15,.0f} /pt")
    print(f"{'=' * 50}")

    # Archive
    if not args.no_archive:
        archive_dir = archive_results(result)
        print(f"  Archived to: {archive_dir}")

    return result


def cmd_synthetic(args):
    """Run pipeline with synthetic data for testing."""
    from pipeline import run_pipeline_synthetic
    from archive import archive_results

    result = run_pipeline_synthetic(
        symbol=args.product,
        underlying_price=args.underlying,
        risk_free_rate=args.rate,
    )

    scores = result["scores"]
    meta = result["metadata"]
    print(f"\n{'=' * 50}")
    print(f"  {args.product} SYNTHETIC @ ${result['underlying_price']:.2f}")
    print(f"  Contracts: {meta['iv_log'].get('converged', 0)}")
    print(f"{'=' * 50}")
    print(f"  GEX:   ${scores['gex']:>15,.0f} /pt")
    print(f"  VEX:   ${scores['vex']:>15,.0f} /pt")
    print(f"  CEX:   ${scores['cex']:>15,.0f} /day")
    print(f"  GEX+:  ${scores['gex_plus']:>15,.0f} /pt")
    print(f"{'=' * 50}")

    # Archive
    if not args.no_archive:
        archive_dir = archive_results(result)
        print(f"  Archived to: {archive_dir}")

    return result


def cmd_purge(args):
    """Purge archived data."""
    from archive import purge

    before = date.fromisoformat(args.before) if args.before else None
    result = purge(
        product=args.product if args.product != "ALL" else None,
        before_date=before,
        tier=args.tier,
        dry_run=not args.force,
    )

    if not args.force:
        print("\n  [DRY RUN] Add --force to actually delete.")
    print(f"  Files: {result['files_count']}, Size: {result['total_mb']} MB")


def cmd_serve(args):
    """Start the dashboard web server."""
    from dashboard import create_app

    app = create_app()
    print(f"\n  Starting dashboard on http://0.0.0.0:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    if args.production:
        # Use gunicorn in production
        import subprocess
        subprocess.run([
            "gunicorn",
            "dashboard:create_app()",
            f"--bind=0.0.0.0:{args.port}",
            f"--workers={args.workers}",
            "--timeout=120",
        ])
    else:
        app.run(debug=args.debug, host="0.0.0.0", port=args.port)


def cmd_validate(args):
    """Run reference case validation."""
    from models import validate_reference_case
    success = validate_reference_case()
    sys.exit(0 if success else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Greek Exposure Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command")

    # ── run ──
    p_run = sub.add_parser("run", help="Run pipeline with live data")
    p_run.add_argument("product", choices=list(PRODUCTS.keys()), help="Product symbol")
    p_run.add_argument("underlying", type=float, help="Current underlying price")
    p_run.add_argument("--date", help="Snapshot date (YYYY-MM-DD, default: today)")
    p_run.add_argument("--rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    p_run.add_argument("--time", default="15:45", help="Snapshot time HH:MM (default: 15:45)")
    p_run.add_argument("--api-key", help="Databento API key (or set DATABENTO_API_KEY env var)")
    p_run.add_argument("--no-archive", action="store_true", help="Skip archiving results")

    # ── synthetic ──
    p_syn = sub.add_parser("synthetic", help="Run with synthetic data (no API key)")
    p_syn.add_argument("--product", default="SPY", choices=list(PRODUCTS.keys()))
    p_syn.add_argument("--underlying", type=float, default=550.0)
    p_syn.add_argument("--rate", type=float, default=0.05)
    p_syn.add_argument("--no-archive", action="store_true", help="Skip archiving results")

    # ── purge ──
    p_purge = sub.add_parser("purge", help="Purge archived data")
    p_purge.add_argument("--product", default="ALL", help="Product or ALL")
    p_purge.add_argument("--before", help="Purge before date (YYYY-MM-DD)")
    p_purge.add_argument("--tier", choices=["1", "2", "all"], default="all")
    p_purge.add_argument("--force", action="store_true", help="Actually delete (default: dry run)")

    # ── serve ──
    p_serve = sub.add_parser("serve", help="Start dashboard server")
    p_serve.add_argument("--port", type=int, default=8050)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.add_argument("--production", action="store_true", help="Use gunicorn")
    p_serve.add_argument("--workers", type=int, default=2)

    # ── validate ──
    sub.add_parser("validate", help="Run reference case validation")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "synthetic": cmd_synthetic,
        "purge": cmd_purge,
        "serve": cmd_serve,
        "validate": cmd_validate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
