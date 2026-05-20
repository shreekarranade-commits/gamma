"""
Exposure aggregation: GEX, VEX, CEX, GEX+.

Takes per-contract Greeks and produces:
  - Headline scores (aggregate)
  - Per-strike profiles
  - Per-strike-per-expiry breakdowns
"""

import numpy as np
import pandas as pd

from config import ProductConfig, EXPIRY_BUCKETS, EXPIRY_BUCKET_LABELS


def compute_sign(is_call: np.ndarray) -> np.ndarray:
    """
    Approach A sign convention:
      Calls: customers sell → dealers long → sign = +1
      Puts:  customers buy  → dealers short → sign = -1
    """
    return np.where(is_call, 1.0, -1.0)


def compute_exposures(
    df: pd.DataFrame,
    underlying_price: float,
    product: ProductConfig,
) -> pd.DataFrame:
    """
    Compute dollar-scaled GEX, VEX, CEX per contract.

    Expects df to have columns:
      strike, expiry, is_call, oi, iv, gamma, vanna, charm, dte

    Adds columns:
      sign, gex, vex, cex

    Parameters
    ----------
    df : DataFrame with per-contract Greeks
    underlying_price : float (spot or futures price)
    product : ProductConfig

    Returns
    -------
    df : DataFrame with exposure columns added
    """
    S = underlying_price
    mult = product.contract_multiplier
    vol_mult = product.vol_spot_multiplier

    sign = compute_sign(df["is_call"].values)

    # GEX: Gamma × OI × Multiplier × sign × S
    # Dollars per 1-point move in underlying
    gex = df["gamma"].values * df["oi"].values * mult * sign * S

    # VEX: Vanna × ΔσPerPoint × OI × Multiplier × sign × S
    # ΔσPerPoint = σ × VolMult / S  (converts spot % move to absolute vol change)
    # After S cancels: Vanna × σ × VolMult × OI × Multiplier × sign
    delta_sigma_per_point = df["iv"].values * vol_mult / S
    vex = df["vanna"].values * delta_sigma_per_point * df["oi"].values * mult * sign * S

    # CEX: -Charm × OI × Multiplier × sign × S / 365
    # Negative because charm = dΔ/dT but time passing means T decreases
    # Dollars per day from time decay
    cex = -df["charm"].values * df["oi"].values * mult * sign * S / 365.0

    df = df.copy()
    df["sign"] = sign
    df["gex"] = gex
    df["vex"] = vex
    df["cex"] = cex

    return df


def compute_headline_scores(df: pd.DataFrame) -> dict:
    """
    Aggregate per-contract exposures into headline scores.

    Returns
    -------
    dict with: gex, vex, cex, gex_plus (all floats)
    """
    gex = df["gex"].sum()
    vex = df["vex"].sum()
    cex = df["cex"].sum()

    return {
        "gex": float(gex),
        "vex": float(vex),
        "cex": float(cex),
        "gex_plus": float(gex + vex),
    }


def compute_strike_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exposures per strike (summed across all expiries).

    Returns DataFrame with columns:
      strike, gex, vex, cex
    """
    profiles = df.groupby("strike").agg(
        gex=("gex", "sum"),
        vex=("vex", "sum"),
        cex=("cex", "sum"),
    ).reset_index()

    return profiles.sort_values("strike")


def assign_expiry_bucket(dte: pd.Series) -> pd.Series:
    """Assign each contract to an expiry bucket based on DTE."""
    bucket = pd.Series("long_term", index=dte.index)
    for name, (lo, hi) in EXPIRY_BUCKETS.items():
        mask = (dte >= lo) & (dte <= hi)
        bucket[mask] = name
    return bucket


def compute_expiry_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exposures per strike per expiry bucket.

    Returns DataFrame with columns:
      strike, expiry_bucket, expiry_label, gex, vex, cex
    """
    df = df.copy()
    df["expiry_bucket"] = assign_expiry_bucket(df["dte"])
    df["expiry_label"] = df["expiry_bucket"].map(EXPIRY_BUCKET_LABELS)

    breakdown = df.groupby(["strike", "expiry_bucket", "expiry_label"]).agg(
        gex=("gex", "sum"),
        vex=("vex", "sum"),
        cex=("cex", "sum"),
    ).reset_index()

    return breakdown.sort_values(["strike", "expiry_bucket"])


def interpret_scores(scores: dict) -> dict:
    """
    Translate raw GEX/VEX/CEX/GEX+ values into regime-aware subtitles
    that consider all three Greeks together rather than each in isolation.

    Returns
    -------
    dict with keys:
      regime    — "STABILIZING" / "FRAGILE" / "DESTABILIZED" / "NEUTRAL"
      gex       — subtitle line for the GEX card
      vex       — subtitle line for the VEX card
      cex       — subtitle line for the CEX card (prepends "OPEX: " when
                  the nearest expiry is <= 2 DTE)
      gex_plus  — subtitle line for the GEX+ card
    """
    gex = float(scores.get("gex", 0) or 0)
    vex = float(scores.get("vex", 0) or 0)
    cex = float(scores.get("cex", 0) or 0)
    gex_plus = float(scores.get("gex_plus", 0) or 0)
    min_dte = scores.get("min_dte")

    # Reference magnitude for "near zero" / "large" judgements on $/pt scores.
    # CEX uses a separate magnitude reference because it lives in $/day units.
    ref = max(abs(gex), abs(vex), abs(gex_plus), 1.0)
    cex_ref = max(abs(cex), 1.0)

    def near_zero(v, r):
        return abs(v) < 0.10 * r

    def is_large(v, r):
        return abs(v) > 0.50 * r

    # ── GEX subtitle (joint with VEX) ─────────────────────────────
    if near_zero(gex, ref):
        gex_sub = "Neutral · No gamma pressure"
    elif gex > 0 and vex > 0:
        gex_sub = "Stabilizing · Dealers absorb moves"
    elif gex > 0 and vex < 0:
        gex_sub = "Stable but fragile · Vol spike risk"
    elif gex < 0 and vex < 0:
        gex_sub = "Destabilized · Moves amplified"
    else:  # gex < 0 and vex >= 0
        gex_sub = "Weak · Gamma negative, vanna supportive"

    # ── VEX subtitle (joint with GEX) ─────────────────────────────
    if vex > 0 and gex > 0:
        vex_sub = "Vol spike = dealer buying · Protected"
    elif vex < 0 and gex > 0:
        vex_sub = "Vol spike = dealer selling · Liquidity cliff below"
    elif vex < 0 and gex < 0:
        vex_sub = "Crash-prone · Both channels destabilize"
    elif vex > 0 and gex < 0:
        vex_sub = "Vanna cushion · Partial protection"
    else:
        vex_sub = ""

    # ── CEX subtitle (with OPEX prefix near expiry) ───────────────
    if near_zero(cex, cex_ref):
        cex_sub = "Minimal time decay pressure"
    elif cex > 0:
        cex_sub = "Sellers tomorrow · Downward drift into OPEX"
    else:
        cex_sub = "Buyers tomorrow · Upward drift into OPEX"
    if min_dte is not None and min_dte <= 2:
        cex_sub = f"OPEX: {cex_sub}"

    # ── GEX+ subtitle ─────────────────────────────────────────────
    if near_zero(gex_plus, ref):
        gex_plus_sub = "Balanced · Options not driving price"
    elif gex_plus > 0 and is_large(gex_plus, ref):
        gex_plus_sub = "Strong liquidity · Range-bound likely"
    elif gex_plus > 0:
        gex_plus_sub = "Mild support · Normal conditions"
    else:
        gex_plus_sub = "Liquidity vacuum · Trend/crash risk"

    # ── Overall regime ────────────────────────────────────────────
    if near_zero(gex, ref) and near_zero(vex, ref):
        regime = "NEUTRAL"
    elif gex > 0 and vex > 0:
        regime = "STABILIZING"
    elif gex < 0 and vex < 0:
        regime = "DESTABILIZED"
    else:
        regime = "FRAGILE"

    return {
        "regime": regime,
        "gex": gex_sub,
        "vex": vex_sub,
        "cex": cex_sub,
        "gex_plus": gex_plus_sub,
    }


def find_flip_strikes(profiles: pd.DataFrame, greek: str) -> list:
    """
    Detect strikes where a Greek's exposure crosses zero.

    For each pair of adjacent strikes whose values have opposite signs,
    interpolate linearly to estimate the zero-crossing strike:

        flip = K_n + (K_{n+1} - K_n) * |v(K_n)| / (|v(K_n)| + |v(K_{n+1})|)

    Returns a list of flip strikes (may be empty, one, or many).
    """
    if profiles is None or len(profiles) < 2 or greek not in profiles.columns:
        return []

    df = profiles.sort_values("strike").reset_index(drop=True)
    strikes = df["strike"].astype(float).values
    values = df[greek].astype(float).values

    flips = []
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        # Skip exact zero anchors at i; treat strict opposite signs as a flip.
        if v0 == 0 and v1 == 0:
            continue
        if v0 * v1 < 0:
            denom = abs(v0) + abs(v1)
            if denom == 0:
                continue
            k = strikes[i] + (strikes[i + 1] - strikes[i]) * abs(v0) / denom
            flips.append(float(k))

    return flips


def build_all_outputs(
    df: pd.DataFrame,
    underlying_price: float,
    product: ProductConfig,
) -> dict:
    """
    Full aggregation pipeline: compute exposures, scores, profiles, breakdowns.

    Parameters
    ----------
    df : DataFrame with per-contract Greeks (gamma, vanna, charm, iv, oi, etc.)
    underlying_price : float
    product : ProductConfig

    Returns
    -------
    dict with keys:
      - contracts: DataFrame with per-contract exposures
      - scores: dict with headline GEX/VEX/CEX/GEX+ and flip strikes
      - strike_profiles: DataFrame with per-strike aggregation
      - expiry_breakdown: DataFrame with per-strike-per-bucket aggregation
    """
    contracts = compute_exposures(df, underlying_price, product)
    scores = compute_headline_scores(contracts)
    profiles = compute_strike_profiles(contracts)
    breakdown = compute_expiry_breakdown(contracts)

    scores["gex_flip"] = find_flip_strikes(profiles, "gex")
    scores["vex_flip"] = find_flip_strikes(profiles, "vex")
    scores["cex_flip"] = find_flip_strikes(profiles, "cex")

    if "dte" in contracts.columns and len(contracts) > 0:
        scores["min_dte"] = int(contracts["dte"].min())
    else:
        scores["min_dte"] = None

    return {
        "contracts": contracts,
        "scores": scores,
        "strike_profiles": profiles,
        "expiry_breakdown": breakdown,
    }
