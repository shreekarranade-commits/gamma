"""
Tests for pipeline.py: synthetic pipeline, filtering, IV solve integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from pipeline import run_pipeline_synthetic, filter_chain
from config import FilterConfig, PRODUCTS


# ══════════════════════════════════════════════════════════════
# 1. SYNTHETIC PIPELINE END-TO-END
# ══════════════════════════════════════════════════════════════

class TestSyntheticPipeline:

    @pytest.fixture(scope="class")
    def spy_result(self):
        """Run synthetic pipeline once for all tests in this class."""
        return run_pipeline_synthetic("SPY", 550.0, 0.05)

    def test_scores_present(self, spy_result):
        for key in ["gex", "vex", "cex", "gex_plus"]:
            assert key in spy_result["scores"]

    def test_scores_are_finite(self, spy_result):
        for key in ["gex", "vex", "cex", "gex_plus"]:
            assert np.isfinite(spy_result["scores"][key]), f"{key} is not finite"

    def test_gex_positive_for_spy(self, spy_result):
        """SPY under normal conditions should have positive GEX (stabilizing)."""
        assert spy_result["scores"]["gex"] > 0

    def test_gex_plus_equals_sum(self, spy_result):
        s = spy_result["scores"]
        assert abs(s["gex_plus"] - (s["gex"] + s["vex"])) < 1.0

    def test_contracts_have_greeks(self, spy_result):
        df = spy_result["contracts"]
        for col in ["delta", "gamma", "vanna", "charm", "iv"]:
            assert col in df.columns
            assert not df[col].isna().all(), f"All {col} values are NaN"

    def test_contracts_have_exposures(self, spy_result):
        df = spy_result["contracts"]
        for col in ["gex", "vex", "cex"]:
            assert col in df.columns

    def test_strike_profiles_nonempty(self, spy_result):
        assert len(spy_result["strike_profiles"]) > 0

    def test_expiry_breakdown_nonempty(self, spy_result):
        assert len(spy_result["expiry_breakdown"]) > 0

    def test_metadata_complete(self, spy_result):
        meta = spy_result["metadata"]
        required = ["product", "underlying_price", "risk_free_rate",
                     "filter_log", "iv_log", "engine_version"]
        for key in required:
            assert key in meta, f"Missing metadata key: {key}"

    def test_iv_convergence_rate_high(self, spy_result):
        """At least 90% of filtered contracts should converge."""
        iv_log = spy_result["metadata"]["iv_log"]
        rate = iv_log["converged"] / iv_log["total"]
        assert rate > 0.90, f"IV convergence rate too low: {rate:.2%}"

    def test_all_products_run(self):
        """Every product in the registry can run synthetic pipeline."""
        for symbol in PRODUCTS:
            if PRODUCTS[symbol].pricing_model.value == "BSM":
                price = 550.0
            else:
                price = 2000.0
            result = run_pipeline_synthetic(symbol, price, 0.05)
            assert result["scores"]["gex"] != 0 or result["scores"]["vex"] != 0, \
                f"All-zero scores for {symbol}"


# ══════════════════════════════════════════════════════════════
# 2. FILTERING
# ══════════════════════════════════════════════════════════════

class TestFiltering:

    @pytest.fixture
    def raw_chain(self):
        """Chain with contracts that should be filtered."""
        from datetime import date as d
        return pd.DataFrame({
            "instrument_id": range(10),
            "strike": [540, 545, 550, 555, 560, 100, 999, 550, 550, 550],
            "expiry": pd.to_datetime([
                "2026-05-01", "2026-05-01", "2026-05-01", "2026-05-01",
                "2026-05-01", "2026-05-01", "2026-05-01", "2026-05-01",
                "2026-04-24", "2026-05-01",  # idx 8: 0 DTE
            ]),
            "is_call": [True] * 10,
            "bid": [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.0, 3.0, 3.0],
            "ask": [5.5, 4.5, 3.5, 2.5, 1.5, 1.0, 1.0, 0.5, 3.5, 3.5],
            "mid_price": [5.25, 4.25, 3.25, 2.25, 1.25, 0.75, 0.75, 0.25, 3.25, 3.25],
            "oi": [1000, 500, 2000, 300, 100, 50, 30, 5, 1000, 1000],
        })

    def test_low_oi_removed(self, raw_chain):
        from datetime import date
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24))
        # OI=5 (idx 7) should be removed
        assert 5 not in filtered["oi"].values

    def test_zero_bid_removed(self, raw_chain):
        from datetime import date
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24))
        # idx 7 has bid=0
        assert log["removed_no_bid"] >= 1

    def test_moneyness_filter(self, raw_chain):
        from datetime import date
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24))
        # Strike 100 and 999 are outside ±20% of 550
        assert 100 not in filtered["strike"].values
        assert 999 not in filtered["strike"].values

    def test_low_dte_removed(self, raw_chain):
        from datetime import date
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24))
        # idx 8: expiry = snapshot date → 0 DTE
        assert log["removed_low_dte"] >= 1

    def test_filter_log_counts(self, raw_chain):
        from datetime import date
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24))
        assert log["start"] == 10
        assert log["after_filter"] == len(filtered)
        assert log["total_removed"] == log["start"] - log["after_filter"]

    def test_custom_filter_thresholds(self, raw_chain):
        """Custom filters are respected."""
        from datetime import date
        strict = FilterConfig(min_open_interest=200)
        filtered, log = filter_chain(raw_chain, 550.0, date(2026, 4, 24), strict)
        # All contracts with OI < 200 should be removed
        assert filtered["oi"].min() >= 200

    def test_empty_after_filter_handled(self):
        """All-filtered chain produces empty DataFrame."""
        from datetime import date
        empty = pd.DataFrame({
            "instrument_id": [0],
            "strike": [100.0],
            "expiry": pd.to_datetime(["2026-05-01"]),
            "is_call": [True],
            "bid": [0.0],
            "ask": [0.5],
            "mid_price": [0.25],
            "oi": [1],
        })
        filtered, log = filter_chain(empty, 550.0, date(2026, 4, 24))
        assert len(filtered) == 0


# ══════════════════════════════════════════════════════════════
# 3. PIPELINE PARAMETERS
# ══════════════════════════════════════════════════════════════

class TestPipelineParameters:

    def test_different_underlying_prices(self):
        """Pipeline produces different scores for different underlying prices."""
        r1 = run_pipeline_synthetic("SPY", 500.0, 0.05)
        r2 = run_pipeline_synthetic("SPY", 600.0, 0.05)
        # Scores should differ meaningfully
        assert r1["scores"]["gex"] != r2["scores"]["gex"]

    def test_different_rates(self):
        """Pipeline produces different scores for different rates."""
        r1 = run_pipeline_synthetic("SPY", 550.0, 0.01)
        r2 = run_pipeline_synthetic("SPY", 550.0, 0.10)
        # Scores should differ (charm is rate-sensitive)
        assert r1["scores"]["cex"] != r2["scores"]["cex"]

    def test_invalid_product_raises(self):
        """Unknown product symbol raises ValueError."""
        with pytest.raises(Exception):
            run_pipeline_synthetic("INVALID", 100.0, 0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
