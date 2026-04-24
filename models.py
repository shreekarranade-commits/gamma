"""
Pricing models and Greek computations.

Implements:
  - Black-Scholes-Merton (BSM) for equity/ETF options
  - Black-76 for futures options
  - Vectorized Newton-Raphson IV solver
  - Gamma, vanna, charm per contract
"""

import numpy as np
from scipy.stats import norm

from config import PricingModel, IVSolverConfig, IV_SOLVER_DEFAULTS


# ══════════════════════════════════════════════════════════════
# STANDARD NORMAL HELPERS
# ══════════════════════════════════════════════════════════════

def _N(x):
    """Cumulative distribution function of standard normal."""
    return norm.cdf(x)


def _phi(x):
    """Probability density function of standard normal."""
    return norm.pdf(x)


# ══════════════════════════════════════════════════════════════
# BSM MODEL (Equities / ETFs)
# ══════════════════════════════════════════════════════════════

def bsm_d1(S, K, T, r, q, sigma):
    """Compute d1 for BSM with continuous dividend yield.

    d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ√T)
    """
    return (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def bsm_d2(d1, sigma, T):
    """Compute d2 from d1."""
    return d1 - sigma * np.sqrt(T)


def bsm_call_price(S, K, T, r, q, sigma):
    """BSM European call price with continuous dividend yield."""
    d1 = bsm_d1(S, K, T, r, q, sigma)
    d2 = bsm_d2(d1, sigma, T)
    return S * np.exp(-q * T) * _N(d1) - K * np.exp(-r * T) * _N(d2)


def bsm_put_price(S, K, T, r, q, sigma):
    """BSM European put price with continuous dividend yield."""
    d1 = bsm_d1(S, K, T, r, q, sigma)
    d2 = bsm_d2(d1, sigma, T)
    return K * np.exp(-r * T) * _N(-d2) - S * np.exp(-q * T) * _N(-d1)


def bsm_price(S, K, T, r, q, sigma, is_call):
    """BSM option price. is_call: boolean array or scalar."""
    call_p = bsm_call_price(S, K, T, r, q, sigma)
    put_p = bsm_put_price(S, K, T, r, q, sigma)
    return np.where(is_call, call_p, put_p)


def bsm_vega(S, K, T, r, q, sigma):
    """BSM vega: dPrice/dSigma. Same for calls and puts."""
    d1 = bsm_d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * _phi(d1) * np.sqrt(T)


def bsm_delta(S, K, T, r, q, sigma, is_call):
    """BSM delta. Call: e^(-qT)*N(d1). Put: -e^(-qT)*N(-d1)."""
    d1 = bsm_d1(S, K, T, r, q, sigma)
    disc = np.exp(-q * T)
    call_delta = disc * _N(d1)
    put_delta = -disc * _N(-d1)
    return np.where(is_call, call_delta, put_delta)


def bsm_gamma(S, K, T, r, q, sigma):
    """BSM gamma: dDelta/dS. Identical for calls and puts."""
    d1 = bsm_d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * _phi(d1) / (S * sigma * np.sqrt(T))


def bsm_vanna(S, K, T, r, q, sigma):
    """BSM vanna: dDelta/dSigma. Sign depends on moneyness via d2.

    vanna = -e^(-qT) · φ(d1) · d2 / σ
    """
    d1 = bsm_d1(S, K, T, r, q, sigma)
    d2 = bsm_d2(d1, sigma, T)
    return -np.exp(-q * T) * _phi(d1) * d2 / sigma


def bsm_charm(S, K, T, r, q, sigma, is_call):
    """
    BSM charm: dDelta/dT (delta decay per unit time).

    Call charm = -q·e^(-qT)·N(d1) + e^(-qT)·φ(d1)·[2(r-q)T - d2·σ√T] / (2T·σ√T)
    Put charm  =  q·e^(-qT)·N(-d1) + e^(-qT)·φ(d1)·[2(r-q)T - d2·σ√T] / (2T·σ√T)

    Returns annualized charm. Divide by 365 for daily.
    """
    d1 = bsm_d1(S, K, T, r, q, sigma)
    d2 = bsm_d2(d1, sigma, T)
    disc = np.exp(-q * T)
    sqrt_T = np.sqrt(T)

    # Common derivative term: e^(-qT) · φ(d1) · ∂d1/∂T
    # ∂d1/∂T = [2(r-q)T - d2·σ√T] / (2T·σ√T)
    dd1_dT = (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    phi_term = disc * _phi(d1) * dd1_dT

    call_charm = -q * disc * _N(d1) + phi_term
    put_charm = q * disc * _N(-d1) + phi_term

    return np.where(is_call, call_charm, put_charm)


# ══════════════════════════════════════════════════════════════
# BLACK-76 MODEL (Futures Options)
# ══════════════════════════════════════════════════════════════

def b76_d1(F, K, T, sigma):
    """Compute d1 for Black-76. Note: r does not appear in d1."""
    return (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def b76_d2(d1, sigma, T):
    """Compute d2 from d1."""
    return d1 - sigma * np.sqrt(T)


def b76_call_price(F, K, T, r, sigma):
    """Black-76 call price."""
    d1 = b76_d1(F, K, T, sigma)
    d2 = b76_d2(d1, sigma, T)
    return np.exp(-r * T) * (F * _N(d1) - K * _N(d2))


def b76_put_price(F, K, T, r, sigma):
    """Black-76 put price."""
    d1 = b76_d1(F, K, T, sigma)
    d2 = b76_d2(d1, sigma, T)
    return np.exp(-r * T) * (K * _N(-d2) - F * _N(-d1))


def b76_price(F, K, T, r, sigma, is_call):
    """Black-76 option price."""
    call_p = b76_call_price(F, K, T, r, sigma)
    put_p = b76_put_price(F, K, T, r, sigma)
    return np.where(is_call, call_p, put_p)


def b76_vega(F, K, T, r, sigma):
    """Black-76 vega."""
    d1 = b76_d1(F, K, T, sigma)
    return np.exp(-r * T) * F * _phi(d1) * np.sqrt(T)


def b76_delta(F, K, T, r, sigma, is_call):
    """Black-76 delta."""
    d1 = b76_d1(F, K, T, sigma)
    disc = np.exp(-r * T)
    call_delta = disc * _N(d1)
    put_delta = -disc * _N(-d1)
    return np.where(is_call, call_delta, put_delta)


def b76_gamma(F, K, T, r, sigma):
    """Black-76 gamma."""
    d1 = b76_d1(F, K, T, sigma)
    return np.exp(-r * T) * _phi(d1) / (F * sigma * np.sqrt(T))


def b76_vanna(F, K, T, r, sigma):
    """Black-76 vanna: dDelta/dSigma.

    vanna = -e^(-rT) · φ(d1) · d2 / σ
    """
    d1 = b76_d1(F, K, T, sigma)
    d2 = b76_d2(d1, sigma, T)
    return -np.exp(-r * T) * _phi(d1) * d2 / sigma


def b76_charm(F, K, T, r, sigma, is_call):
    """
    Black-76 charm: dDelta/dT.

    Call charm = -r·e^(-rT)·N(d1) - e^(-rT)·φ(d1)·d2/(2T)
    Put charm  =  r·e^(-rT)·N(-d1) - e^(-rT)·φ(d1)·d2/(2T)
    """
    d1 = b76_d1(F, K, T, sigma)
    d2 = b76_d2(d1, sigma, T)
    disc = np.exp(-r * T)

    # Common term: e^(-rT) · φ(d1) · dd1/dT where dd1/dT = -d2/(2T)
    phi_term = -disc * _phi(d1) * d2 / (2 * T)

    call_charm = -r * disc * _N(d1) + phi_term
    put_charm = r * disc * _N(-d1) + phi_term

    return np.where(is_call, call_charm, put_charm)


# ══════════════════════════════════════════════════════════════
# UNIFIED GREEK INTERFACE
# ══════════════════════════════════════════════════════════════

def compute_greeks(S, K, T, r, sigma, is_call, model, q=0.0):
    """
    Compute delta, gamma, vanna, charm for a set of contracts.

    Parameters
    ----------
    S : float or array - Underlying price (spot for BSM, futures for B76)
    K : array - Strike prices
    T : array - Time to expiry in years
    r : float - Risk-free rate
    sigma : array - Implied volatility per contract
    is_call : boolean array - True for calls
    model : PricingModel - BSM or BLACK76
    q : float - Continuous dividend yield (BSM only)

    Returns
    -------
    dict with keys: delta, gamma, vanna, charm (all arrays)
    """
    if model == PricingModel.BSM:
        return {
            "delta": bsm_delta(S, K, T, r, q, sigma, is_call),
            "gamma": bsm_gamma(S, K, T, r, q, sigma),
            "vanna": bsm_vanna(S, K, T, r, q, sigma),
            "charm": bsm_charm(S, K, T, r, q, sigma, is_call),
        }
    elif model == PricingModel.BLACK76:
        return {
            "delta": b76_delta(S, K, T, r, sigma, is_call),
            "gamma": b76_gamma(S, K, T, r, sigma),
            "vanna": b76_vanna(S, K, T, r, sigma),
            "charm": b76_charm(S, K, T, r, sigma, is_call),
        }
    else:
        raise ValueError(f"Unknown pricing model: {model}")


# ══════════════════════════════════════════════════════════════
# VECTORIZED NEWTON-RAPHSON IV SOLVER
# ══════════════════════════════════════════════════════════════

def solve_iv(
    market_price,
    S, K, T, r, is_call,
    model,
    q=0.0,
    config=IV_SOLVER_DEFAULTS,
):
    """
    Solve for implied volatility using Newton-Raphson.

    Parameters
    ----------
    market_price : array - Observed option mid prices
    S : float or array - Underlying price
    K : array - Strikes
    T : array - Time to expiry (years)
    r : float - Risk-free rate
    is_call : boolean array
    model : PricingModel
    q : float - Dividend yield (BSM only)
    config : IVSolverConfig

    Returns
    -------
    iv : array - Implied volatilities (NaN where non-convergent)
    converged : boolean array - True where solver converged
    """
    n = len(market_price)
    sigma = np.full(n, config.initial_guess, dtype=np.float64)
    converged = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    # Pre-filter: exclude negative prices or prices below intrinsic
    if model == PricingModel.BSM:
        call_intrinsic = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        put_intrinsic = np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    else:
        call_intrinsic = np.maximum(np.exp(-r * T) * (S - K), 0)
        put_intrinsic = np.maximum(np.exp(-r * T) * (K - S), 0)

    intrinsic = np.where(is_call, call_intrinsic, put_intrinsic)
    bad_price = (market_price <= 0) | (market_price < intrinsic * 0.95)
    active[bad_price] = False

    for iteration in range(config.max_iterations):
        if not np.any(active):
            break

        idx = active

        # Compute model price and vega at current sigma
        if model == PricingModel.BSM:
            model_price = bsm_price(S, K[idx], T[idx], r, q, sigma[idx], is_call[idx])
            vega = bsm_vega(S, K[idx], T[idx], r, q, sigma[idx])
        else:
            model_price = b76_price(S, K[idx], T[idx], r, sigma[idx], is_call[idx])
            vega = b76_vega(S, K[idx], T[idx], r, sigma[idx])

        # Newton-Raphson update
        price_diff = model_price - market_price[idx]
        vega_safe = np.where(np.abs(vega) < 1e-12, 1e-12, vega)
        sigma_update = sigma[idx] - price_diff / vega_safe

        # Clamp to bounds
        sigma_update = np.clip(sigma_update, config.lower_bound, config.upper_bound)

        # Check convergence
        change = np.abs(sigma_update - sigma[idx])
        newly_converged = change < config.tolerance

        # Update state
        conv_mask = np.zeros(n, dtype=bool)
        conv_mask[idx] = newly_converged
        converged |= conv_mask

        sigma[idx] = sigma_update

        # Remove converged from active set
        active_indices = np.where(active)[0]
        active[active_indices[newly_converged]] = False

        # Also deactivate if vega is too small (deep ITM/OTM)
        tiny_vega = np.abs(vega) < 1e-10
        if np.any(tiny_vega):
            deactivate = active_indices[tiny_vega & ~newly_converged]
            active[deactivate] = False

    # Mark non-converged as NaN
    sigma[~converged] = np.nan

    return sigma, converged


# ══════════════════════════════════════════════════════════════
# VALIDATION / REFERENCE CASE
# ══════════════════════════════════════════════════════════════

def validate_reference_case():
    """
    Run the reference case from the functional spec (Section 12.1)
    and verify against expected values.

    Setup: OTM Put, S=3000, K=2900, T=30/365, sigma=0.20, r=0, q=0
    Expected: delta≈0.2676, gamma≈0.001914, vanna≈0.9261 (unsigned OTM put)
    """
    S = 3000.0
    K = np.array([2900.0])
    T = np.array([30.0 / 365.0])
    r = 0.0
    q = 0.0
    sigma = np.array([0.20])
    is_call = np.array([False])

    greeks = compute_greeks(S, K, T, r, sigma, is_call, PricingModel.BSM, q)

    results = {
        "d1": bsm_d1(S, K[0], T[0], r, q, sigma[0]),
        "d2": bsm_d2(bsm_d1(S, K[0], T[0], r, q, sigma[0]), sigma[0], T[0]),
        "phi_d1": _phi(bsm_d1(S, K[0], T[0], r, q, sigma[0])),
        "put_delta_unsigned": abs(greeks["delta"][0]),
        "gamma": greeks["gamma"][0],
        "vanna_raw": greeks["vanna"][0],
        "charm_annualized": greeks["charm"][0],
        "charm_daily": greeks["charm"][0] / 365,
    }

    expected = {
        "d1": 0.6199,
        "d2": 0.5626,
        "phi_d1": 0.3292,
        "put_delta_unsigned": 0.2676,
        "gamma": 0.001914,
    }

    print("=" * 60)
    print("REFERENCE CASE VALIDATION")
    print("OTM Put: S=3000, K=2900, T=30/365, σ=0.20, r=0, q=0")
    print("=" * 60)

    all_pass = True
    for key in expected:
        actual = results[key]
        exp = expected[key]
        match = abs(actual - exp) < 0.001
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  {key:25s}: {actual:10.4f}  (expected {exp:.4f})  [{status}]")

    print(f"\n  vanna (raw dΔ/dσ):        {results['vanna_raw']:10.6f}")
    print(f"  charm (annualized):       {results['charm_annualized']:10.6f}")
    print(f"  charm (daily):            {results['charm_daily']:10.6f}")

    # Vanna sign flip test: ITM put (S=2800)
    S_itm = 2800.0
    d1_itm = bsm_d1(S_itm, K[0], T[0], r, q, sigma[0])
    d2_itm = bsm_d2(d1_itm, sigma[0], T[0])
    greeks_itm = compute_greeks(S_itm, K, T, r, sigma, is_call, PricingModel.BSM, q)

    print(f"\n  --- Vanna Sign Flip Test ---")
    print(f"  OTM (S=3000): d2 = {results['d2']:+.4f}, vanna = {results['vanna_raw']:+.6f}")
    print(f"  ITM (S=2800): d2 = {d2_itm:+.4f}, vanna = {greeks_itm['vanna'][0]:+.6f}")

    otm_sign = np.sign(results["vanna_raw"])
    itm_sign = np.sign(greeks_itm["vanna"][0])
    flip = otm_sign != itm_sign
    print(f"  Sign flip detected: {flip}  [{'PASS' if flip else 'FAIL'}]")
    if not flip:
        all_pass = False

    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'=' * 60}")

    return all_pass


if __name__ == "__main__":
    validate_reference_case()
