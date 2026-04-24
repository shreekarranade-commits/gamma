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
      - scores: dict with headline GEX/VEX/CEX/GEX+
      - strike_profiles: DataFrame with per-strike aggregation
      - expiry_breakdown: DataFrame with per-strike-per-bucket aggregation
    """
    contracts = compute_exposures(df, underlying_price, product)
    scores = compute_headline_scores(contracts)
    profiles = compute_strike_profiles(contracts)
    breakdown = compute_expiry_breakdown(contracts)

    return {
        "contracts": contracts,
        "scores": scores,
        "strike_profiles": profiles,
        "expiry_breakdown": breakdown,
    }
