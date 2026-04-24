"""Live Databento end-to-end test: GC (gold) futures options via Black-76.

Exercises the GLBX.MDP3 path (mbp-1 quotes, Black-76 pricing, European style).
Front-month futures price is inferred from the chain via put-call parity on
futures:  C - P = exp(-rT) * (F - K)  →  F = K + (C - P) * exp(rT)  at ATM.
"""
import logging
import math
import os
from datetime import date

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("test_live_gc")

from config import PRODUCTS
from pipeline import ingest_chain, build_chain_dataframe, run_pipeline


SNAPSHOT = date(2026, 4, 23)
RISK_FREE_RATE = 0.05


def infer_futures_price(chain: pd.DataFrame, snapshot: date, r: float) -> tuple[float, int]:
    """Black-76 parity at ATM of the nearest expiry with both calls and puts."""
    df = chain.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["dte"] = (df["expiry"] - pd.Timestamp(snapshot)).dt.days
    df = df[df["dte"] >= 1]

    for dte in sorted(df["dte"].unique()):
        rows = df[df["dte"] == dte]
        calls = rows[rows["is_call"]][["strike", "mid_price"]].rename(columns={"mid_price": "c"})
        puts = rows[~rows["is_call"]][["strike", "mid_price"]].rename(columns={"mid_price": "p"})
        paired = calls.merge(puts, on="strike")
        if len(paired) == 0:
            continue
        paired["diff"] = (paired["c"] - paired["p"]).abs()
        atm = paired.loc[paired["diff"].idxmin()]
        K = float(atm["strike"])
        T = dte / 365.0
        F = K + (float(atm["c"]) - float(atm["p"])) * math.exp(r * T)
        log.info(
            f"Parity on {dte} DTE: K*={K:.2f}, C={atm['c']:.3f}, P={atm['p']:.3f} "
            f"→ F={F:.3f}"
        )
        return F, dte
    raise RuntimeError("No call/put pairs found on any expiry")


def main():
    product = PRODUCTS["GC"]
    log.info(f"Key prefix: {os.environ['DATABENTO_API_KEY'][:5]}..., "
             f"len={len(os.environ['DATABENTO_API_KEY'])}")
    log.info(f"Product: {product.symbol}  dataset={product.dataset}  "
             f"model={product.pricing_model.value}  mult={product.contract_multiplier}")

    raw = ingest_chain(product, SNAPSHOT, snapshot_time="13:15")  # COMEX gold closes ~13:30 ET
    for name, df in raw.items():
        log.info(f"  {name}: {len(df)} rows")

    chain = build_chain_dataframe(raw, product)
    log.info(f"Raw chain: {len(chain)} contracts")

    F, dte_used = infer_futures_price(chain, SNAPSHOT, RISK_FREE_RATE)
    log.info(f"Front-month GC futures ≈ {F:.2f} (from {dte_used} DTE parity)")

    log.info("Running full pipeline…")
    result = run_pipeline(
        symbol="GC",
        snapshot_date=SNAPSHOT,
        underlying_price=F,
        risk_free_rate=RISK_FREE_RATE,
        snapshot_time="13:15",
    )

    scores = result["scores"]
    meta = result["metadata"]
    print("\n" + "=" * 68)
    print(f"LIVE GC PIPELINE  |  {SNAPSHOT}  |  F≈{F:.2f}  (Black-76)")
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

    assert len(result["contracts"]) > 0, "No contracts survived pipeline"
    assert math.isfinite(scores["gex"]), "GEX is non-finite"
    assert meta["iv_log"]["converged"] > 0, "IV solver converged on zero contracts"
    log.info("All sanity checks passed.")


if __name__ == "__main__":
    main()
