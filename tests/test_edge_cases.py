"""
Tests for edge cases and numerical stability.

Covers:
  - Extreme moneyness (deep ITM/OTM)
  - Very short / very long expiry
  - Very high / very low volatility
  - Zero dividend yield
  - Large arrays (performance smoke test)
  - NaN/inf handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import time
from numpy.testing import assert_allclose

from models import (
    bsm_call_price, bsm_put_price, bsm_gamma, bsm_vanna, bsm_charm,
    bsm_delta, bsm_vega, compute_greeks, solve_iv,
    b76_call_price, b76_gamma, b76_vanna,
)
from config import PricingModel, IVSolverConfig
from pipeline import run_pipeline_synthetic


# ══════════════════════════════════════════════════════════════
# 1. EXTREME MONEYNESS
# ══════════════════════════════════════════════════════════════

class TestExtremeMoneyness:

    def test_deep_otm_call_price_near_zero(self):
        """50% OTM call should be nearly zero."""
        price = bsm_call_price(100.0, 150.0, 30/365, 0.05, 0.0, 0.20)
        assert price < 0.001

    def test_deep_itm_call_price_near_intrinsic(self):
        """50% ITM call should be near intrinsic."""
        S, K, T, r, q, sigma = 150.0, 100.0, 30/365, 0.05, 0.0, 0.20
        price = bsm_call_price(S, K, T, r, q, sigma)
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(price - intrinsic) < 0.5

    def test_deep_otm_gamma_near_zero(self):
        gamma = bsm_gamma(100.0, 200.0, 30/365, 0.05, 0.0, 0.20)
        assert gamma < 1e-10

    def test_deep_otm_delta_near_zero(self):
        delta = bsm_delta(100.0, 200.0, 30/365, 0.05, 0.0, 0.20, True)
        assert abs(delta) < 0.001

    def test_deep_itm_delta_near_one(self):
        delta = bsm_delta(200.0, 100.0, 30/365, 0.05, 0.0, 0.20, True)
        assert delta > 0.99

    def test_greeks_finite_at_extremes(self):
        """Greeks should be finite even at extreme moneyness."""
        K = np.array([10.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
        T = np.full(6, 30/365)
        sigma = np.full(6, 0.20)
        is_call = np.full(6, True)
        greeks = compute_greeks(100.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.0)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} has non-finite values at extreme moneyness"


# ══════════════════════════════════════════════════════════════
# 2. EXTREME TIME TO EXPIRY
# ══════════════════════════════════════════════════════════════

class TestExtremeExpiry:

    def test_very_short_expiry_greeks_finite(self):
        """1-day option should produce finite Greeks."""
        K = np.array([550.0])
        T = np.array([1/365])
        sigma = np.array([0.18])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.013)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} not finite at 1-day expiry"

    def test_very_long_expiry_greeks_finite(self):
        """2-year option should produce finite Greeks."""
        K = np.array([550.0])
        T = np.array([730/365])
        sigma = np.array([0.25])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.013)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} not finite at 2-year expiry"

    def test_gamma_explodes_near_expiry_atm(self):
        """ATM gamma should be much larger at 1 DTE than 30 DTE."""
        g_30 = bsm_gamma(550.0, 550.0, 30/365, 0.05, 0.013, 0.18)
        g_1 = bsm_gamma(550.0, 550.0, 1/365, 0.05, 0.013, 0.18)
        assert g_1 > g_30 * 3  # should be much larger


# ══════════════════════════════════════════════════════════════
# 3. EXTREME VOLATILITY
# ══════════════════════════════════════════════════════════════

class TestExtremeVolatility:

    def test_very_low_vol_greeks_finite(self):
        """1% vol should produce finite Greeks."""
        K = np.array([550.0])
        T = np.array([30/365])
        sigma = np.array([0.01])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.013)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} not finite at 1% vol"

    def test_very_high_vol_greeks_finite(self):
        """300% vol (meme stock) should produce finite Greeks."""
        K = np.array([200.0])
        T = np.array([30/365])
        sigma = np.array([3.00])
        is_call = np.array([True])
        greeks = compute_greeks(200.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.0)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} not finite at 300% vol"

    def test_high_vol_iv_solver_converges(self):
        """IV solver should handle 200% vol."""
        S, K, T, r, q, true_iv = 200.0, 200.0, 30/365, 0.05, 0.0, 2.00
        price = bsm_call_price(S, K, T, r, q, true_iv)
        iv, conv = solve_iv(
            np.array([price]), S, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BSM, q
        )
        assert conv[0], f"Failed to converge for 200% vol, price={price}"
        assert_allclose(iv[0], true_iv, atol=0.05)

    def test_gamma_decreases_with_higher_vol(self):
        """Higher vol → lower gamma (wider distribution = less delta sensitivity)."""
        g_low = bsm_gamma(550.0, 550.0, 30/365, 0.05, 0.013, 0.10)
        g_high = bsm_gamma(550.0, 550.0, 30/365, 0.05, 0.013, 0.50)
        assert g_low > g_high


# ══════════════════════════════════════════════════════════════
# 4. ZERO / EDGE PARAMETER VALUES
# ══════════════════════════════════════════════════════════════

class TestZeroParameters:

    def test_zero_dividend_yield(self):
        """q=0 should work (TSLA case)."""
        K = np.array([550.0])
        T = np.array([30/365])
        sigma = np.array([0.18])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.0)
        assert np.all(np.isfinite(greeks["gamma"]))

    def test_zero_rate(self):
        """r=0 should work (original paper assumption)."""
        K = np.array([550.0])
        T = np.array([30/365])
        sigma = np.array([0.18])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, 0.0, sigma, is_call, PricingModel.BSM, 0.0)
        assert np.all(np.isfinite(greeks["gamma"]))

    def test_negative_rate(self):
        """Negative r should produce finite results."""
        K = np.array([550.0])
        T = np.array([30/365])
        sigma = np.array([0.18])
        is_call = np.array([True])
        greeks = compute_greeks(550.0, K, T, -0.01, sigma, is_call, PricingModel.BSM, 0.0)
        for name, vals in greeks.items():
            assert np.all(np.isfinite(vals)), f"{name} not finite with negative rate"


# ══════════════════════════════════════════════════════════════
# 5. IV SOLVER EDGE CASES
# ══════════════════════════════════════════════════════════════

class TestIVSolverEdgeCases:

    def test_below_intrinsic_returns_nan(self):
        """Price below intrinsic → non-convergent."""
        S, K, T, r, q = 550.0, 500.0, 30/365, 0.05, 0.013
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        bad_price = intrinsic * 0.5  # below intrinsic
        iv, conv = solve_iv(
            np.array([bad_price]), S, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BSM, q
        )
        assert not conv[0]

    def test_mixed_convergence(self):
        """Mix of good and bad prices: good converge, bad don't."""
        S = 550.0
        K = np.array([550.0, 550.0, 550.0])
        T = np.full(3, 30/365)
        r, q = 0.05, 0.013
        true_iv = 0.18

        good_price = bsm_call_price(S, K[0], T[0], r, q, true_iv)
        prices = np.array([good_price, -1.0, 0.0])
        is_call = np.array([True, True, True])

        iv, conv = solve_iv(prices, S, K, T, r, is_call, PricingModel.BSM, q)
        assert conv[0] is True or conv[0] == True
        assert not conv[1]
        assert not conv[2]

    def test_custom_solver_config(self):
        """Custom solver parameters are respected."""
        S, K, T, r, q, true_iv = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        price = bsm_call_price(S, K, T, r, q, true_iv)

        strict = IVSolverConfig(tolerance=1e-10, max_iterations=200)
        iv, conv = solve_iv(
            np.array([price]), S, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BSM, q, strict
        )
        assert conv[0]
        assert_allclose(iv[0], true_iv, atol=1e-8)


# ══════════════════════════════════════════════════════════════
# 6. PERFORMANCE SMOKE TEST
# ══════════════════════════════════════════════════════════════

class TestPerformance:

    def test_large_chain_iv_solver(self):
        """10,000 contracts should solve IV in under 2 seconds."""
        n = 10000
        S = 550.0
        K = np.random.uniform(450, 650, n)
        T = np.random.uniform(1/365, 90/365, n)
        true_iv = np.random.uniform(0.10, 0.40, n)
        is_call = np.random.choice([True, False], n)
        r, q = 0.05, 0.013

        prices = np.array([
            bsm_call_price(S, K[i], T[i], r, q, true_iv[i]) if is_call[i]
            else bsm_put_price(S, K[i], T[i], r, q, true_iv[i])
            for i in range(n)
        ])

        t0 = time.time()
        iv, conv = solve_iv(prices, S, K, T, r, is_call, PricingModel.BSM, q)
        elapsed = time.time() - t0

        conv_rate = conv.sum() / n
        assert elapsed < 2.0, f"IV solver too slow: {elapsed:.2f}s for {n} contracts"
        assert conv_rate > 0.95, f"Low convergence rate: {conv_rate:.2%}"

    def test_large_chain_greeks(self):
        """10,000 Greek computations should be near-instant."""
        n = 10000
        S = 550.0
        K = np.random.uniform(450, 650, n)
        T = np.random.uniform(1/365, 90/365, n)
        sigma = np.random.uniform(0.10, 0.40, n)
        is_call = np.random.choice([True, False], n)

        t0 = time.time()
        greeks = compute_greeks(S, K, T, 0.05, sigma, is_call, PricingModel.BSM, 0.013)
        elapsed = time.time() - t0

        assert elapsed < 0.5, f"Greek computation too slow: {elapsed:.2f}s"
        assert len(greeks["gamma"]) == n

    def test_synthetic_pipeline_under_5_seconds(self):
        """Full synthetic pipeline should complete in under 5 seconds."""
        t0 = time.time()
        result = run_pipeline_synthetic("SPY", 550.0, 0.05)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Pipeline too slow: {elapsed:.2f}s"


# ══════════════════════════════════════════════════════════════
# 7. DETERMINISM
# ══════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_inputs_same_outputs(self):
        """Pipeline is deterministic: same inputs → same scores."""
        r1 = run_pipeline_synthetic("SPY", 550.0, 0.05)
        r2 = run_pipeline_synthetic("SPY", 550.0, 0.05)
        assert_allclose(r1["scores"]["gex"], r2["scores"]["gex"])
        assert_allclose(r1["scores"]["vex"], r2["scores"]["vex"])
        assert_allclose(r1["scores"]["cex"], r2["scores"]["cex"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
