"""
Tests for models.py: BSM pricing, Black-76 pricing, Greeks, IV solver.

Covers:
  - Reference case from functional spec
  - Put-call parity
  - Greek symmetries and boundary conditions
  - IV solver convergence and round-trip accuracy
  - Black-76 model parity with BSM when q=0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from models import (
    bsm_d1, bsm_d2, bsm_call_price, bsm_put_price, bsm_price,
    bsm_vega, bsm_delta, bsm_gamma, bsm_vanna, bsm_charm,
    b76_d1, b76_d2, b76_call_price, b76_put_price, b76_price,
    b76_vega, b76_delta, b76_gamma, b76_vanna, b76_charm,
    compute_greeks, solve_iv, validate_reference_case,
    _N, _phi,
)
from config import PricingModel


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def ref_case():
    """Reference case from spec: OTM put, S=3000, K=2900."""
    return dict(S=3000.0, K=2900.0, T=30/365, r=0.0, q=0.0, sigma=0.20)


@pytest.fixture
def spy_atm():
    """ATM SPY call, realistic parameters."""
    return dict(S=550.0, K=550.0, T=30/365, r=0.05, q=0.013, sigma=0.18)


@pytest.fixture
def futures_case():
    """Gold futures option: /GC."""
    return dict(F=2350.0, K=2400.0, T=60/365, r=0.05, sigma=0.15)


# ══════════════════════════════════════════════════════════════
# 1. REFERENCE CASE VALIDATION
# ══════════════════════════════════════════════════════════════

class TestReferenceCase:
    """Tests against the spec reference case (Section 12.1)."""

    def test_d1(self, ref_case):
        d1 = bsm_d1(ref_case["S"], ref_case["K"], ref_case["T"],
                     ref_case["r"], ref_case["q"], ref_case["sigma"])
        assert_allclose(d1, 0.6199, atol=0.001)

    def test_d2(self, ref_case):
        d1 = bsm_d1(ref_case["S"], ref_case["K"], ref_case["T"],
                     ref_case["r"], ref_case["q"], ref_case["sigma"])
        d2 = bsm_d2(d1, ref_case["sigma"], ref_case["T"])
        assert_allclose(d2, 0.5626, atol=0.001)

    def test_put_delta(self, ref_case):
        delta = bsm_delta(ref_case["S"], ref_case["K"], ref_case["T"],
                          ref_case["r"], ref_case["q"], ref_case["sigma"], False)
        assert_allclose(abs(delta), 0.2676, atol=0.001)

    def test_gamma(self, ref_case):
        gamma = bsm_gamma(ref_case["S"], ref_case["K"], ref_case["T"],
                          ref_case["r"], ref_case["q"], ref_case["sigma"])
        assert_allclose(gamma, 0.001914, atol=0.0001)

    def test_charm_daily(self, ref_case):
        charm = bsm_charm(ref_case["S"], ref_case["K"], ref_case["T"],
                          ref_case["r"], ref_case["q"], ref_case["sigma"], False)
        charm_daily = charm / 365
        # dDelta/dT for OTM put is negative (delta decays toward 0 as T decreases)
        assert_allclose(abs(charm_daily), 0.003087, atol=0.0001)

    def test_full_validation_passes(self):
        assert validate_reference_case() is True


# ══════════════════════════════════════════════════════════════
# 2. BSM PRICING
# ══════════════════════════════════════════════════════════════

class TestBSMPricing:
    """BSM call/put pricing tests."""

    def test_atm_call_positive(self, spy_atm):
        p = bsm_call_price(**{k: spy_atm[k] for k in ["S", "K", "T", "r", "q", "sigma"]})
        assert p > 0

    def test_atm_put_positive(self, spy_atm):
        p = bsm_put_price(spy_atm["S"], spy_atm["K"], spy_atm["T"],
                          spy_atm["r"], spy_atm["q"], spy_atm["sigma"])
        assert p > 0

    def test_put_call_parity(self, spy_atm):
        """C - P = S*e^(-qT) - K*e^(-rT)"""
        S, K, T, r, q, sigma = [spy_atm[k] for k in ["S", "K", "T", "r", "q", "sigma"]]
        call = bsm_call_price(S, K, T, r, q, sigma)
        put = bsm_put_price(S, K, T, r, q, sigma)
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert_allclose(call - put, expected, rtol=1e-10)

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call price ≈ S*e^(-qT) - K*e^(-rT)."""
        S, K, T, r, q, sigma = 550.0, 400.0, 30/365, 0.05, 0.013, 0.18
        price = bsm_call_price(S, K, T, r, q, sigma)
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert price >= intrinsic * 0.99

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call price → 0."""
        price = bsm_call_price(550.0, 800.0, 30/365, 0.05, 0.013, 0.18)
        assert price < 0.01

    def test_price_increases_with_vol(self):
        """Higher vol → higher option price."""
        S, K, T, r, q = 550.0, 550.0, 30/365, 0.05, 0.013
        p_low = bsm_call_price(S, K, T, r, q, 0.10)
        p_high = bsm_call_price(S, K, T, r, q, 0.30)
        assert p_high > p_low

    def test_price_increases_with_time(self):
        """More time → higher option price (all else equal)."""
        S, K, r, q, sigma = 550.0, 550.0, 0.05, 0.013, 0.18
        p_short = bsm_call_price(S, K, 7/365, r, q, sigma)
        p_long = bsm_call_price(S, K, 90/365, r, q, sigma)
        assert p_long > p_short

    def test_vectorized_pricing(self):
        """Pricing works with arrays."""
        S = 550.0
        K = np.array([500, 525, 550, 575, 600])
        T = np.full(5, 30/365)
        is_call = np.array([True, True, True, True, True])
        prices = bsm_price(S, K, T, 0.05, 0.013, np.full(5, 0.18), is_call)
        assert prices.shape == (5,)
        # Prices should decrease as strike increases for calls
        assert np.all(np.diff(prices) < 0)


# ══════════════════════════════════════════════════════════════
# 3. BSM GREEKS
# ══════════════════════════════════════════════════════════════

class TestBSMGreeks:
    """BSM Greek computation tests."""

    def test_call_delta_between_0_and_1(self):
        S, K, T, r, q, sigma = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        delta = bsm_delta(S, K, T, r, q, sigma, True)
        assert 0 < delta < 1

    def test_put_delta_between_neg1_and_0(self):
        S, K, T, r, q, sigma = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        delta = bsm_delta(S, K, T, r, q, sigma, False)
        assert -1 < delta < 0

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = e^(-qT)."""
        S, K, T, r, q, sigma = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        call_d = bsm_delta(S, K, T, r, q, sigma, True)
        put_d = bsm_delta(S, K, T, r, q, sigma, False)
        assert_allclose(call_d - put_d, np.exp(-q * T), rtol=1e-10)

    def test_gamma_always_positive(self):
        S, K, T, r, q, sigma = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        gamma = bsm_gamma(S, K, T, r, q, sigma)
        assert gamma > 0

    def test_gamma_same_for_call_and_put(self):
        """Gamma is identical for calls and puts (it's not option-type dependent)."""
        K = np.array([550.0])
        T = np.array([30/365])
        sigma = np.array([0.18])
        greeks_call = compute_greeks(550.0, K, T, 0.05, sigma, np.array([True]), PricingModel.BSM, 0.013)
        greeks_put = compute_greeks(550.0, K, T, 0.05, sigma, np.array([False]), PricingModel.BSM, 0.013)
        assert_allclose(greeks_call["gamma"], greeks_put["gamma"], rtol=1e-10)

    def test_gamma_peaks_at_atm(self):
        """Gamma is highest ATM."""
        S = 550.0
        strikes = np.array([500, 525, 550, 575, 600], dtype=float)
        T = np.full(5, 30/365)
        sigma = np.full(5, 0.18)
        gammas = bsm_gamma(S, strikes, T, 0.05, 0.013, sigma)
        atm_idx = 2  # strike=550
        assert gammas[atm_idx] == gammas.max()

    def test_gamma_increases_near_expiry(self):
        """ATM gamma increases as expiry approaches."""
        S, K, r, q, sigma = 550.0, 550.0, 0.05, 0.013, 0.18
        g_30d = bsm_gamma(S, K, 30/365, r, q, sigma)
        g_7d = bsm_gamma(S, K, 7/365, r, q, sigma)
        g_2d = bsm_gamma(S, K, 2/365, r, q, sigma)
        assert g_2d > g_7d > g_30d

    def test_vega_positive(self):
        vega = bsm_vega(550.0, 550.0, 30/365, 0.05, 0.013, 0.18)
        assert vega > 0

    def test_vanna_sign_flip_otm_to_itm(self):
        """Vanna flips sign when option goes from OTM to ITM."""
        K = np.array([2900.0])
        T = np.array([30/365])
        sigma = np.array([0.20])

        # OTM put (S > K)
        vanna_otm = bsm_vanna(3000.0, K[0], T[0], 0.0, 0.0, sigma[0])
        # ITM put (S < K)
        vanna_itm = bsm_vanna(2800.0, K[0], T[0], 0.0, 0.0, sigma[0])

        assert np.sign(vanna_otm) != np.sign(vanna_itm), \
            f"Vanna should flip sign: OTM={vanna_otm}, ITM={vanna_itm}"


# ══════════════════════════════════════════════════════════════
# 4. BSM GREEKS - NUMERICAL DIFFERENTIATION VERIFICATION
# ══════════════════════════════════════════════════════════════

class TestGreeksNumerical:
    """Verify analytical Greeks match numerical (finite-difference) derivatives."""

    def _num_delta(self, S, K, T, r, q, sigma, is_call, dS=0.01):
        p_up = bsm_price(S + dS, K, T, r, q, sigma, is_call)
        p_dn = bsm_price(S - dS, K, T, r, q, sigma, is_call)
        return (p_up - p_dn) / (2 * dS)

    def _num_gamma(self, S, K, T, r, q, sigma, is_call, dS=0.01):
        d_up = self._num_delta(S + dS, K, T, r, q, sigma, is_call)
        d_dn = self._num_delta(S - dS, K, T, r, q, sigma, is_call)
        return (d_up - d_dn) / (2 * dS)

    def _num_vanna(self, S, K, T, r, q, sigma, is_call, dSigma=0.001):
        d_up = bsm_delta(S, K, T, r, q, sigma + dSigma, is_call)
        d_dn = bsm_delta(S, K, T, r, q, sigma - dSigma, is_call)
        return (d_up - d_dn) / (2 * dSigma)

    def _num_charm(self, S, K, T, r, q, sigma, is_call, dT=1e-5):
        d_up = bsm_delta(S, K, T + dT, r, q, sigma, is_call)
        d_dn = bsm_delta(S, K, T - dT, r, q, sigma, is_call)
        return (d_up - d_dn) / (2 * dT)

    @pytest.mark.parametrize("K,is_call", [
        (500.0, True), (550.0, True), (600.0, True),
        (500.0, False), (550.0, False), (600.0, False),
    ])
    def test_delta_matches_numerical(self, K, is_call):
        S, T, r, q, sigma = 550.0, 30/365, 0.05, 0.013, 0.18
        analytical = bsm_delta(S, K, T, r, q, sigma, is_call)
        numerical = self._num_delta(S, K, T, r, q, sigma, is_call)
        assert_allclose(analytical, numerical, atol=1e-4)

    @pytest.mark.parametrize("K", [500.0, 550.0, 600.0])
    def test_gamma_matches_numerical(self, K):
        S, T, r, q, sigma = 550.0, 30/365, 0.05, 0.013, 0.18
        analytical = bsm_gamma(S, K, T, r, q, sigma)
        numerical = self._num_gamma(S, K, T, r, q, sigma, True)
        assert_allclose(analytical, numerical, rtol=0.01)

    @pytest.mark.parametrize("K", [500.0, 550.0, 600.0])
    def test_vanna_matches_numerical(self, K):
        S, T, r, q, sigma = 550.0, 30/365, 0.05, 0.013, 0.18
        analytical = bsm_vanna(S, K, T, r, q, sigma)
        numerical = self._num_vanna(S, K, T, r, q, sigma, True)
        assert_allclose(analytical, numerical, rtol=0.02)

    @pytest.mark.parametrize("K,is_call", [
        (500.0, True), (550.0, True), (600.0, True),
        (500.0, False), (550.0, False),
    ])
    def test_charm_matches_numerical(self, K, is_call):
        S, T, r, q, sigma = 550.0, 30/365, 0.05, 0.013, 0.18
        analytical = bsm_charm(S, K, T, r, q, sigma, is_call)
        numerical = self._num_charm(S, K, T, r, q, sigma, is_call)
        assert_allclose(analytical, numerical, rtol=0.05)


# ══════════════════════════════════════════════════════════════
# 5. BLACK-76 MODEL
# ══════════════════════════════════════════════════════════════

class TestBlack76:
    """Black-76 pricing and Greeks for futures options."""

    def test_put_call_parity(self, futures_case):
        F, K, T, r, sigma = [futures_case[k] for k in ["F", "K", "T", "r", "sigma"]]
        call = b76_call_price(F, K, T, r, sigma)
        put = b76_put_price(F, K, T, r, sigma)
        expected = np.exp(-r * T) * (F - K)
        assert_allclose(call - put, expected, rtol=1e-10)

    def test_d1_no_rate_dependency(self, futures_case):
        """Black-76 d1 does not depend on r."""
        F, K, T, sigma = [futures_case[k] for k in ["F", "K", "T", "sigma"]]
        d1_r0 = b76_d1(F, K, T, sigma)
        d1_r5 = b76_d1(F, K, T, sigma)  # same call, r not a parameter
        assert_allclose(d1_r0, d1_r5)

    def test_gamma_positive(self, futures_case):
        F, K, T, r, sigma = [futures_case[k] for k in ["F", "K", "T", "r", "sigma"]]
        gamma = b76_gamma(F, K, T, r, sigma)
        assert gamma > 0

    def test_call_delta_between_0_and_1(self, futures_case):
        F, K, T, r, sigma = [futures_case[k] for k in ["F", "K", "T", "r", "sigma"]]
        delta = b76_delta(F, K, T, r, sigma, True)
        assert 0 < delta < 1

    def test_greeks_via_unified_interface(self, futures_case):
        """compute_greeks works with BLACK76 model."""
        K = np.array([futures_case["K"]])
        T = np.array([futures_case["T"]])
        sigma = np.array([futures_case["sigma"]])
        is_call = np.array([True])
        greeks = compute_greeks(
            futures_case["F"], K, T, futures_case["r"],
            sigma, is_call, PricingModel.BLACK76
        )
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vanna" in greeks
        assert "charm" in greeks
        assert greeks["gamma"][0] > 0

    def test_b76_converges_to_bsm_when_q0_and_F_equals_Se_rT(self):
        """When F = S*e^(rT) and q=0, B76 and BSM should produce same prices."""
        S, K, T, r, sigma = 550.0, 550.0, 30/365, 0.05, 0.18
        F = S * np.exp(r * T)
        bsm_c = bsm_call_price(S, K, T, r, 0.0, sigma)
        b76_c = b76_call_price(F, K, T, r, sigma)
        assert_allclose(bsm_c, b76_c, rtol=1e-8)


# ══════════════════════════════════════════════════════════════
# 6. IV SOLVER
# ══════════════════════════════════════════════════════════════

class TestIVSolver:
    """Implied volatility solver tests."""

    def test_round_trip_atm_call(self):
        """Price → IV → price round trip for ATM call."""
        S, K, T, r, q, true_iv = 550.0, 550.0, 30/365, 0.05, 0.013, 0.18
        price = bsm_call_price(S, K, T, r, q, true_iv)
        iv, conv = solve_iv(
            np.array([price]), S, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BSM, q
        )
        assert conv[0], "IV solver did not converge"
        assert_allclose(iv[0], true_iv, atol=1e-5)

    def test_round_trip_otm_put(self):
        """Price → IV → price round trip for OTM put."""
        S, K, T, r, q, true_iv = 550.0, 500.0, 30/365, 0.05, 0.013, 0.22
        price = bsm_put_price(S, K, T, r, q, true_iv)
        iv, conv = solve_iv(
            np.array([price]), S, np.array([K]), np.array([T]),
            r, np.array([False]), PricingModel.BSM, q
        )
        assert conv[0]
        assert_allclose(iv[0], true_iv, atol=1e-5)

    def test_round_trip_vectorized(self):
        """Vectorized round trip across multiple contracts."""
        S = 550.0
        K = np.array([500, 520, 540, 550, 560, 580, 600], dtype=float)
        T = np.full(7, 30/365)
        r, q = 0.05, 0.013
        true_iv = np.array([0.22, 0.20, 0.19, 0.18, 0.19, 0.21, 0.24])
        is_call = np.array([False, False, False, True, True, True, True])

        prices = bsm_price(S, K, T, r, q, true_iv, is_call)
        iv, conv = solve_iv(prices, S, K, T, r, is_call, PricingModel.BSM, q)

        assert np.all(conv), f"Failures at indices: {np.where(~conv)[0]}"
        assert_allclose(iv, true_iv, atol=1e-4)

    def test_round_trip_black76(self):
        """Round trip for futures option."""
        F, K, T, r, true_iv = 2350.0, 2400.0, 60/365, 0.05, 0.15
        price = b76_call_price(F, K, T, r, true_iv)
        iv, conv = solve_iv(
            np.array([price]), F, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BLACK76
        )
        assert conv[0]
        assert_allclose(iv[0], true_iv, atol=1e-5)

    def test_high_iv_convergence(self):
        """Solver converges for high IV options (meme stocks)."""
        S, K, T, r, q, true_iv = 200.0, 200.0, 7/365, 0.05, 0.0, 1.50
        price = bsm_call_price(S, K, T, r, q, true_iv)
        iv, conv = solve_iv(
            np.array([price]), S, np.array([K]), np.array([T]),
            r, np.array([True]), PricingModel.BSM, q
        )
        assert conv[0]
        assert_allclose(iv[0], true_iv, atol=0.01)

    def test_negative_price_returns_nan(self):
        """Negative market price → non-convergent (NaN)."""
        iv, conv = solve_iv(
            np.array([-1.0]), 550.0, np.array([550.0]), np.array([30/365]),
            0.05, np.array([True]), PricingModel.BSM, 0.013
        )
        assert not conv[0]
        assert np.isnan(iv[0])

    def test_zero_price_returns_nan(self):
        """Zero market price → non-convergent."""
        iv, conv = solve_iv(
            np.array([0.0]), 550.0, np.array([550.0]), np.array([30/365]),
            0.05, np.array([True]), PricingModel.BSM, 0.013
        )
        assert not conv[0]


# ══════════════════════════════════════════════════════════════
# 7. NORMAL DISTRIBUTION HELPERS
# ══════════════════════════════════════════════════════════════

class TestNormalHelpers:

    def test_N_symmetry(self):
        assert_allclose(_N(0), 0.5)

    def test_N_bounds(self):
        assert _N(-10) < 1e-10
        assert _N(10) > 1 - 1e-10

    def test_phi_symmetry(self):
        assert_allclose(_phi(1.0), _phi(-1.0))

    def test_phi_peak(self):
        assert _phi(0) > _phi(1) > _phi(2) > _phi(3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
