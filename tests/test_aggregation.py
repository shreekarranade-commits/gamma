"""
Tests for aggregation.py: sign convention, dollar scaling, profiles, breakdowns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from aggregation import (
    compute_sign, compute_exposures, compute_headline_scores,
    compute_strike_profiles, assign_expiry_bucket, compute_expiry_breakdown,
    build_all_outputs,
)
from config import PRODUCTS


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_contracts():
    """Minimal contract DataFrame for testing."""
    return pd.DataFrame({
        "strike": [540.0, 540.0, 550.0, 550.0, 560.0, 560.0],
        "expiry": pd.to_datetime(["2026-05-01"] * 6),
        "is_call": [True, False, True, False, True, False],
        "oi": [1000, 2000, 5000, 8000, 3000, 1500],
        "iv": [0.20, 0.22, 0.18, 0.19, 0.21, 0.23],
        "gamma": [0.005, 0.005, 0.010, 0.010, 0.004, 0.004],
        "vanna": [-0.001, -0.001, -0.0005, -0.0005, 0.001, 0.001],
        "charm": [0.50, 0.60, 1.20, 1.30, 0.40, 0.45],
        "dte": [7, 7, 7, 7, 7, 7],
    })


@pytest.fixture
def multi_expiry_contracts():
    """Contracts across multiple expiry buckets."""
    rows = []
    for dte_val in [1, 5, 14, 45]:
        for is_call in [True, False]:
            rows.append({
                "strike": 550.0,
                "expiry": pd.Timestamp("2026-04-24") + pd.Timedelta(days=dte_val),
                "is_call": is_call,
                "oi": 1000,
                "iv": 0.18,
                "gamma": 0.01,
                "vanna": -0.001,
                "charm": 1.0,
                "dte": dte_val,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 1. SIGN CONVENTION
# ══════════════════════════════════════════════════════════════

class TestSignConvention:

    def test_calls_positive(self):
        is_call = np.array([True, True, True])
        signs = compute_sign(is_call)
        assert np.all(signs == 1.0)

    def test_puts_negative(self):
        is_call = np.array([False, False, False])
        signs = compute_sign(is_call)
        assert np.all(signs == -1.0)

    def test_mixed(self):
        is_call = np.array([True, False, True, False])
        signs = compute_sign(is_call)
        expected = np.array([1.0, -1.0, 1.0, -1.0])
        assert_allclose(signs, expected)


# ══════════════════════════════════════════════════════════════
# 2. EXPOSURE COMPUTATION
# ══════════════════════════════════════════════════════════════

class TestExposureComputation:

    def test_gex_formula(self, sample_contracts):
        """GEX = gamma × OI × multiplier × sign × S."""
        product = PRODUCTS["SPY"]
        S = 550.0
        result = compute_exposures(sample_contracts, S, product)

        # Check first row: call, sign=+1
        row = result.iloc[0]
        expected_gex = row["gamma"] * row["oi"] * 100 * 1.0 * S
        assert_allclose(row["gex"], expected_gex)

        # Check second row: put, sign=-1
        row = result.iloc[1]
        expected_gex = row["gamma"] * row["oi"] * 100 * (-1.0) * S
        assert_allclose(row["gex"], expected_gex)

    def test_cex_daily_division(self, sample_contracts):
        """CEX includes -charm/365 for daily conversion (negated because T decreases)."""
        product = PRODUCTS["SPY"]
        S = 550.0
        result = compute_exposures(sample_contracts, S, product)

        row = result.iloc[0]
        expected_cex = -row["charm"] * row["oi"] * 100 * 1.0 * S / 365
        assert_allclose(row["cex"], expected_cex)

    def test_vex_includes_vol_multiplier(self, sample_contracts):
        """VEX uses the vol-spot multiplier from product config."""
        product = PRODUCTS["SPY"]
        S = 550.0
        result = compute_exposures(sample_contracts, S, product)

        row = result.iloc[0]
        delta_sigma = row["iv"] * product.vol_spot_multiplier / S
        expected_vex = row["vanna"] * delta_sigma * row["oi"] * 100 * 1.0 * S
        assert_allclose(row["vex"], expected_vex)

    def test_different_multiplier_for_futures(self, sample_contracts):
        """Futures products use their own contract multiplier."""
        product_spy = PRODUCTS["SPY"]
        product_cl = PRODUCTS["CL"]
        S = 70.0  # crude price

        result_spy = compute_exposures(sample_contracts.copy(), S, product_spy)
        result_cl = compute_exposures(sample_contracts.copy(), S, product_cl)

        # CL multiplier is 1000, SPY is 100, so CL GEX should be 10x
        ratio = result_cl["gex"].abs().sum() / result_spy["gex"].abs().sum()
        assert_allclose(ratio, 10.0, rtol=0.01)

    def test_sign_column_added(self, sample_contracts):
        product = PRODUCTS["SPY"]
        result = compute_exposures(sample_contracts, 550.0, product)
        assert "sign" in result.columns
        assert "gex" in result.columns
        assert "vex" in result.columns
        assert "cex" in result.columns


# ══════════════════════════════════════════════════════════════
# 3. HEADLINE SCORES
# ══════════════════════════════════════════════════════════════

class TestHeadlineScores:

    def test_scores_are_sums(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        scores = compute_headline_scores(exposed)

        assert_allclose(scores["gex"], exposed["gex"].sum())
        assert_allclose(scores["vex"], exposed["vex"].sum())
        assert_allclose(scores["cex"], exposed["cex"].sum())

    def test_gex_plus_is_gex_plus_vex(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        scores = compute_headline_scores(exposed)
        assert_allclose(scores["gex_plus"], scores["gex"] + scores["vex"])

    def test_scores_are_floats(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        scores = compute_headline_scores(exposed)
        for key in ["gex", "vex", "cex", "gex_plus"]:
            assert isinstance(scores[key], float)


# ══════════════════════════════════════════════════════════════
# 4. STRIKE PROFILES
# ══════════════════════════════════════════════════════════════

class TestStrikeProfiles:

    def test_one_row_per_strike(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        profiles = compute_strike_profiles(exposed)
        assert len(profiles) == 3  # 540, 550, 560

    def test_sorted_by_strike(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        profiles = compute_strike_profiles(exposed)
        assert profiles["strike"].is_monotonic_increasing

    def test_profile_sums_match_headline(self, sample_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(sample_contracts, 550.0, product)
        scores = compute_headline_scores(exposed)
        profiles = compute_strike_profiles(exposed)
        assert_allclose(profiles["gex"].sum(), scores["gex"], rtol=1e-10)
        assert_allclose(profiles["vex"].sum(), scores["vex"], rtol=1e-10)
        assert_allclose(profiles["cex"].sum(), scores["cex"], rtol=1e-10)


# ══════════════════════════════════════════════════════════════
# 5. EXPIRY BUCKETS
# ══════════════════════════════════════════════════════════════

class TestExpiryBuckets:

    def test_bucket_assignment(self):
        dte = pd.Series([0, 1, 2, 3, 5, 7, 8, 14, 30, 31, 60, 90])
        buckets = assign_expiry_bucket(dte)
        assert buckets.iloc[0] == "near_term"     # 0 DTE
        assert buckets.iloc[2] == "near_term"     # 2 DTE
        assert buckets.iloc[3] == "short_term"    # 3 DTE
        assert buckets.iloc[5] == "short_term"    # 7 DTE
        assert buckets.iloc[6] == "medium_term"   # 8 DTE
        assert buckets.iloc[8] == "medium_term"   # 30 DTE
        assert buckets.iloc[9] == "long_term"     # 31 DTE

    def test_breakdown_has_all_buckets(self, multi_expiry_contracts):
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(multi_expiry_contracts, 550.0, product)
        breakdown = compute_expiry_breakdown(exposed)
        buckets = breakdown["expiry_bucket"].unique()
        assert "near_term" in buckets
        assert "short_term" in buckets
        assert "medium_term" in buckets
        assert "long_term" in buckets

    def test_breakdown_sums_to_profile(self, multi_expiry_contracts):
        """Sum of all buckets per strike == strike profile total."""
        product = PRODUCTS["SPY"]
        exposed = compute_exposures(multi_expiry_contracts, 550.0, product)
        profiles = compute_strike_profiles(exposed)
        breakdown = compute_expiry_breakdown(exposed)

        strike_sum = breakdown.groupby("strike")["gex"].sum()
        for strike in profiles["strike"]:
            profile_val = profiles[profiles["strike"] == strike]["gex"].iloc[0]
            breakdown_val = strike_sum.get(strike, 0)
            assert_allclose(profile_val, breakdown_val, rtol=1e-10)


# ══════════════════════════════════════════════════════════════
# 6. FULL OUTPUT BUILDER
# ══════════════════════════════════════════════════════════════

class TestBuildAllOutputs:

    def test_output_keys(self, sample_contracts):
        product = PRODUCTS["SPY"]
        result = build_all_outputs(sample_contracts, 550.0, product)
        assert "contracts" in result
        assert "scores" in result
        assert "strike_profiles" in result
        assert "expiry_breakdown" in result

    def test_contracts_have_exposure_columns(self, sample_contracts):
        product = PRODUCTS["SPY"]
        result = build_all_outputs(sample_contracts, 550.0, product)
        for col in ["gex", "vex", "cex", "sign"]:
            assert col in result["contracts"].columns

    def test_empty_dataframe_handled(self):
        """Empty input produces zero scores."""
        product = PRODUCTS["SPY"]
        empty = pd.DataFrame(columns=[
            "strike", "expiry", "is_call", "oi", "iv",
            "gamma", "vanna", "charm", "dte"
        ])
        result = build_all_outputs(empty, 550.0, product)
        assert result["scores"]["gex"] == 0
        assert result["scores"]["vex"] == 0
        assert result["scores"]["cex"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
