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


# ══════════════════════════════════════════════════════════════
# 4. VOLUME CAPTURE (Enhancement 1, v1.3)
# ══════════════════════════════════════════════════════════════

class TestVolumeCapture:

    @pytest.fixture(scope="class")
    def result(self):
        return run_pipeline_synthetic("SPY", 550.0, 0.05)

    def test_volume_column_present(self, result):
        """Volume column flows through to contracts DataFrame."""
        assert "volume" in result["contracts"].columns

    def test_volume_to_oi_ratio_present(self, result):
        """volume_to_oi_ratio computed column flows through."""
        assert "volume_to_oi_ratio" in result["contracts"].columns

    def test_volume_nonnegative(self, result):
        """Volume values are non-negative."""
        assert (result["contracts"]["volume"] >= 0).all()

    def test_volume_to_oi_ratio_consistent(self, result):
        """volume_to_oi_ratio == volume / max(oi, 1) within tolerance."""
        df = result["contracts"]
        expected = df["volume"] / df["oi"].clip(lower=1)
        np.testing.assert_allclose(df["volume_to_oi_ratio"].values, expected.values, rtol=1e-9)


# ══════════════════════════════════════════════════════════════
# 5. PUT-CALL PARITY SPOT INFERENCE (Enhancement 3, v1.3)
# ══════════════════════════════════════════════════════════════

class TestSpotInference:

    def _build_synthetic_chain(self, S=550.0, r=0.05, q=0.013, dte=30,
                                model="BSM"):
        """Build a clean synthetic chain whose mid prices satisfy parity exactly."""
        from datetime import date as d
        from models import bsm_call_price, bsm_put_price, b76_call_price, b76_put_price
        snapshot = d(2026, 4, 24)
        expiry = pd.Timestamp(snapshot) + pd.Timedelta(days=dte)
        T = dte / 365.0
        sigma = 0.20

        rows = []
        for K in np.arange(S - 25, S + 26, 5.0):
            if model == "BSM":
                c = bsm_call_price(S, K, T, r, q, sigma)
                p = bsm_put_price(S, K, T, r, q, sigma)
            else:
                c = b76_call_price(S, K, T, r, sigma)
                p = b76_put_price(S, K, T, r, sigma)
            for is_call, mid in [(True, float(c)), (False, float(p))]:
                rows.append({
                    "instrument_id": len(rows),
                    "strike": float(K),
                    "expiry": expiry,
                    "is_call": is_call,
                    "bid": max(mid * 0.99, 0.01),
                    "ask": mid * 1.01,
                    "mid_price": mid,
                    "oi": 1000,
                    "volume": 100,
                    "volume_to_oi_ratio": 0.1,
                })
        return pd.DataFrame(rows), snapshot

    def test_bsm_spot_recovered(self):
        """BSM parity recovers spot within 0.1%."""
        from pipeline import infer_underlying_price
        S_true = 550.0
        r, q = 0.05, 0.013
        chain, snapshot = self._build_synthetic_chain(S=S_true, r=r, q=q, model="BSM")
        S_inferred = infer_underlying_price(chain, snapshot, PRODUCTS["SPY"], r)
        assert abs(S_inferred - S_true) / S_true < 0.001, \
            f"Inferred {S_inferred} vs true {S_true}"

    def test_black76_futures_recovered(self):
        """Black-76 parity recovers futures price within 0.1%."""
        from pipeline import infer_underlying_price
        F_true = 2400.0
        r = 0.05
        chain, snapshot = self._build_synthetic_chain(S=F_true, r=r, model="BLACK76")
        F_inferred = infer_underlying_price(chain, snapshot, PRODUCTS["GC"], r)
        assert abs(F_inferred - F_true) / F_true < 0.001, \
            f"Inferred {F_inferred} vs true {F_true}"

    def test_no_pairs_raises(self):
        """If no call/put pairs exist, ValueError is raised."""
        from pipeline import infer_underlying_price
        from datetime import date as d
        snapshot = d(2026, 4, 24)
        chain = pd.DataFrame({
            "instrument_id": [0, 1],
            "strike": [550.0, 555.0],
            "expiry": [pd.Timestamp(snapshot) + pd.Timedelta(days=30)] * 2,
            "is_call": [True, True],
            "bid": [5.0, 4.0],
            "ask": [5.5, 4.5],
            "mid_price": [5.25, 4.25],
            "oi": [1000, 1000],
        })
        with pytest.raises(ValueError):
            infer_underlying_price(chain, snapshot, PRODUCTS["SPY"], 0.05)


# ══════════════════════════════════════════════════════════════
# 6. DATABENTO QUERY-WINDOW CLAMPING (T+0 publish-lag handling)
# ══════════════════════════════════════════════════════════════

class TestQueryWindowClamping:
    """
    Databento historical feeds lag real-time and don't fully materialize
    until T+1, so a naive `start=today, end=tomorrow` query produces 422
    errors. resolve_query_window inspects metadata.get_dataset_range and
    (a) falls back to the most recent published date when today isn't
    there yet, and (b) clamps every query end to the available high-water
    mark.
    """

    def test_today_unpublished_falls_back(self):
        """OPRA case: today not yet published → snapshot_date falls back."""
        from datetime import date, datetime, timezone
        from pipeline import resolve_query_window
        # Today = Apr 28, but feed end is Apr 27 22:00 UTC.
        avail_end = datetime(2026, 4, 27, 22, 0, tzinfo=timezone.utc)
        w = resolve_query_window(
            avail_end=avail_end,
            dataset="OPRA.PILLAR",
            snapshot_date=date(2026, 4, 28),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 27)
        # def_start should now be Apr 27, def_end clamped to avail_end.
        assert w["def_start"] == "2026-04-27"
        # The naive next_date "2026-04-28" would parse as Apr 28 00:00 UTC,
        # which is past avail_end (Apr 27 22:00) — must be clamped.
        assert pd.Timestamp(w["def_end"]) <= pd.Timestamp(avail_end)
        # Quote window at 15:55 UTC on Apr 27 is well inside the available
        # range; should not be clamped.
        assert w["quote_start"].startswith("2026-04-27T15:55")
        assert w["quote_end"].startswith("2026-04-27T15:55")

    def test_avail_end_at_midnight_boundary_falls_back(self):
        """
        Edge case: avail_end is exactly midnight UTC of the requested date
        (e.g. OPRA's daily-close summary record posted at 00:00:00 of T+1).
        The requested calendar day contains no data inside its interior, so
        we must still fall back to the previous trading day — not stay on
        the requested date and produce a degenerate start==end query.
        """
        from datetime import date, datetime, timezone
        from pipeline import resolve_query_window
        avail_end = datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc)
        w = resolve_query_window(
            avail_end=avail_end,
            dataset="OPRA.PILLAR",
            snapshot_date=date(2026, 4, 28),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 27)
        assert w["def_start"] == "2026-04-27"
        # def_end clamped to the boundary (Apr 28 00:00 UTC) — start < end.
        de = pd.Timestamp(w["def_end"])
        if de.tz is None:
            de = de.tz_localize("UTC")
        assert pd.Timestamp("2026-04-27", tz="UTC") < de

    def test_today_published_date_kept_end_clamped(self):
        """GLBX case: today partially published → keep date, clamp ends."""
        from datetime import date, datetime, timezone
        from pipeline import resolve_query_window
        # Apr 28, available up to 19:00 UTC same day.
        avail_end = datetime(2026, 4, 28, 19, 0, tzinfo=timezone.utc)
        w = resolve_query_window(
            avail_end=avail_end,
            dataset="GLBX.MDP3",
            snapshot_date=date(2026, 4, 28),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 28)
        # def_end would naively be Apr 29 00:00 — must clamp to avail_end.
        assert pd.Timestamp(w["def_end"]) <= pd.Timestamp(avail_end)
        assert pd.Timestamp(w["def_end"]) == pd.Timestamp(avail_end)
        # GLBX stats window 00:00–02:00 fits inside available; not clamped.
        assert w["stats_start"] == "2026-04-28T00:00"
        assert w["stats_end"] == "2026-04-28T02:00"

    def test_snapshot_time_past_avail_end_steps_back_to_previous_day(self):
        """
        If snapshot_time on today doesn't fit (e.g. tick fires at 11:55 ET
        but the OPRA feed has only published through 09:30 ET so far), the
        helper must step the date back to a fully-published trading day —
        not slide the window into the empty pre-open seconds.
        """
        from datetime import date, datetime, timezone
        from pipeline import resolve_query_window
        # Apr 28 2026 is a Tuesday; Apr 27 is Monday (a trading day).
        # Snapshot 15:55 UTC requested, but only 13:30 UTC published today.
        avail_end = datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc)
        w = resolve_query_window(
            avail_end=avail_end,
            dataset="OPRA.PILLAR",
            snapshot_date=date(2026, 4, 28),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 27)
        # Quote window lands inside Apr 27 well within the published range.
        assert w["quote_start"].startswith("2026-04-27T15:55")

    def test_step_back_skips_weekends(self):
        """Step-back walks past Sat/Sun to the previous Friday."""
        from datetime import date, datetime, timezone
        from pipeline import resolve_query_window
        # Mon 2026-04-27, partial early-session publish; previous trading
        # day is Fri 2026-04-24 (Sun Apr 26 and Sat Apr 25 are weekends).
        avail_end = datetime(2026, 4, 27, 13, 30, tzinfo=timezone.utc)
        w = resolve_query_window(
            avail_end=avail_end,
            dataset="OPRA.PILLAR",
            snapshot_date=date(2026, 4, 27),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 24)
        assert date(2026, 4, 24).weekday() == 4  # Friday

    def test_no_avail_end_passes_through(self):
        """If metadata fetch fails (avail_end=None) we don't clamp anything."""
        from datetime import date
        from pipeline import resolve_query_window
        w = resolve_query_window(
            avail_end=None,
            dataset="OPRA.PILLAR",
            snapshot_date=date(2026, 4, 28),
            snapshot_time="15:55",
            snapshot_window_seconds=1,
        )
        assert w["effective_date"] == date(2026, 4, 28)
        assert w["def_start"] == "2026-04-28"
        assert w["def_end"] == "2026-04-29"

    def test_get_dataset_available_end_parses_response(self):
        """_get_dataset_available_end parses Databento's metadata reply."""
        from unittest.mock import MagicMock
        from pipeline import _get_dataset_available_end
        client = MagicMock()
        client.metadata.get_dataset_range.return_value = {
            "start": "2013-04-01T00:00:00.000000000Z",
            "end": "2026-04-27T22:00:00.000000000Z",
        }
        end = _get_dataset_available_end(client, "OPRA.PILLAR")
        assert end is not None
        assert end.year == 2026 and end.month == 4 and end.day == 27
        assert end.hour == 22
        # Returned datetime is UTC-aware.
        assert end.tzinfo is not None

    def test_get_dataset_available_end_swallows_errors(self):
        """If the metadata call raises, helper returns None (no clamping)."""
        from unittest.mock import MagicMock
        from pipeline import _get_dataset_available_end
        client = MagicMock()
        client.metadata.get_dataset_range.side_effect = RuntimeError("network down")
        assert _get_dataset_available_end(client, "OPRA.PILLAR") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
