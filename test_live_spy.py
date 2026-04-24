"""Live Databento end-to-end test: SPY chain → filter → IV → Greeks → aggregates.

Spot is inferred from the chain via put-call parity at the ATM strike of the
nearest listed expiry: F = K* + C(K*) - P(K*), where K* minimizes |C - P|.
"""
import logging
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("test_live_spy")

from config import PRODUCTS
from pipeline import ingest_chain, build_chain_dataframe, run_pipeline


SNAPSHOT = date(2026, 4, 23)
RISK_FREE_RATE = 0.05


def infer_spot_from_chain(chain: pd.DataFrame, snapshot: date, r: float, q: float) -> float:
    """Put-call parity at ATM of the nearest expiry.

    Forward F ≈ K* + C(K*) − P(K*) at the strike K* where |C − P| is minimal.
    Spot S = F * exp(-(r - q) * T).
    """
    df = chain.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["dte"] = (df["expiry"] - pd.Timestamp(snapshot)).dt.days
    df = df[df["dte"] >= 1]

    near_dte = df["dte"].min()
    near = df[df["dte"] == near_dte]
    log.info(f"Parity: nearest expiry is {near_dte} DTE ({len(near)} contracts)")

    calls = near[near["is_call"]][["strike", "mid_price"]].rename(columns={"mid_price": "call_mid"})
    puts = near[~near["is_call"]][["strike", "mid_price"]].rename(columns={"mid_price": "put_mid"})
    paired = calls.merge(puts, on="strike")
    if len(paired) == 0:
        raise RuntimeError("No call/put pairs found on nearest expiry")

    paired["diff"] = (paired["call_mid"] - paired["put_mid"]).abs()
    atm = paired.loc[paired["diff"].idxmin()]
    K_star = float(atm["strike"])
    F = K_star + float(atm["call_mid"]) - float(atm["put_mid"])
    T = near_dte / 365.0
    S = F * math.exp(-(r - q) * T)
    log.info(
        f"Parity: ATM strike K*={K_star:.2f}, C={atm['call_mid']:.3f}, "
        f"P={atm['put_mid']:.3f}, F={F:.3f}, T={T:.4f}y → S≈{S:.3f}"
    )
    return S


def main():
    product = PRODUCTS["SPY"]
    q = product.dividend_yield or 0.0
    log.info(f"Key prefix: {os.environ['DATABENTO_API_KEY'][:5]}..., "
             f"len={len(os.environ['DATABENTO_API_KEY'])}")

    # ── Stage 1: ingest + build raw chain (also used for spot inference) ──
    raw = ingest_chain(product, SNAPSHOT, snapshot_time="15:45")
    chain = build_chain_dataframe(raw, product)
    log.info(f"Raw chain: {len(chain)} contracts")

    spot = infer_spot_from_chain(chain, SNAPSHOT, RISK_FREE_RATE, q)

    # ── Stage 2: full pipeline ──
    log.info("Running full pipeline…")
    try:
        result = run_pipeline(
            symbol="SPY",
            snapshot_date=SNAPSHOT,
            underlying_price=spot,
            risk_free_rate=RISK_FREE_RATE,
        )
    except Exception as e:
        log.error(f"run_pipeline failed: {type(e).__name__}: {e}")
        raise

    scores = result["scores"]
    meta = result["metadata"]
    print("\n" + "=" * 68)
    print(f"LIVE SPY PIPELINE  |  {SNAPSHOT}  |  spot={spot:.2f}")
    print("=" * 68)
    print(f"  Contracts (post-filter, IV-converged): {len(result['contracts'])}")
    print(f"  Strike profiles rows : {len(result['strike_profiles'])}")
    print(f"  Expiry breakdown rows: {len(result['expiry_breakdown'])}")
    print(f"\n  ── Headline Scores ──")
    print(f"  GEX   : ${scores['gex']:>18,.0f} /pt")
    print(f"  VEX   : ${scores['vex']:>18,.0f} /pt")
    print(f"  CEX   : ${scores['cex']:>18,.0f} /day")
    print(f"  GEX+  : ${scores['gex_plus']:>18,.0f} /pt")
    print(f"\n  Filter log : {meta['filter_log']}")
    print(f"  IV log     : {meta['iv_log']}")
    print(f"  Total time : {meta['total_time_seconds']}s")
    print("=" * 68)

    # Basic sanity checks
    assert len(result["contracts"]) > 0, "No contracts survived pipeline"
    assert math.isfinite(scores["gex"]), "GEX is non-finite"
    assert math.isfinite(scores["vex"]), "VEX is non-finite"
    assert meta["iv_log"]["converged"] > 0, "IV solver converged on zero contracts"
    log.info("All sanity checks passed.")


if __name__ == "__main__":
    main()
