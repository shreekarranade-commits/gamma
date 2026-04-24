"""
Product configuration registry and global constants.
Each product defines its data source, pricing model, and parameters.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PricingModel(Enum):
    BSM = "BSM"
    BLACK76 = "BLACK76"


class ExerciseStyle(Enum):
    AMERICAN = "AMERICAN"
    EUROPEAN = "EUROPEAN"


class UnderlyingSource(Enum):
    EQUITY_SPOT = "EQUITY_SPOT"
    FUTURES_PRICE = "FUTURES_PRICE"


@dataclass
class ProductConfig:
    symbol: str
    dataset: str
    parent_symbol: str
    pricing_model: PricingModel
    contract_multiplier: int
    dividend_yield: Optional[float]
    underlying_source: UnderlyingSource
    vol_spot_multiplier: float
    exercise_style: ExerciseStyle


# ── Product Registry ──────────────────────────────────────────

PRODUCTS = {
    "SPY": ProductConfig(
        symbol="SPY",
        dataset="OPRA.PILLAR",
        parent_symbol="SPY.OPT",
        pricing_model=PricingModel.BSM,
        contract_multiplier=100,
        dividend_yield=0.013,
        underlying_source=UnderlyingSource.EQUITY_SPOT,
        vol_spot_multiplier=10.0,
        exercise_style=ExerciseStyle.AMERICAN,
    ),
    "QQQ": ProductConfig(
        symbol="QQQ",
        dataset="OPRA.PILLAR",
        parent_symbol="QQQ.OPT",
        pricing_model=PricingModel.BSM,
        contract_multiplier=100,
        dividend_yield=0.006,
        underlying_source=UnderlyingSource.EQUITY_SPOT,
        vol_spot_multiplier=10.0,
        exercise_style=ExerciseStyle.AMERICAN,
    ),
    "TSLA": ProductConfig(
        symbol="TSLA",
        dataset="OPRA.PILLAR",
        parent_symbol="TSLA.OPT",
        pricing_model=PricingModel.BSM,
        contract_multiplier=100,
        dividend_yield=0.0,
        underlying_source=UnderlyingSource.EQUITY_SPOT,
        vol_spot_multiplier=15.0,
        exercise_style=ExerciseStyle.AMERICAN,
    ),
    "GC": ProductConfig(
        symbol="GC",
        dataset="GLBX.MDP3",
        parent_symbol="OG.OPT",
        pricing_model=PricingModel.BLACK76,
        contract_multiplier=100,
        dividend_yield=None,
        underlying_source=UnderlyingSource.FUTURES_PRICE,
        vol_spot_multiplier=5.0,
        exercise_style=ExerciseStyle.EUROPEAN,
    ),
    "CL": ProductConfig(
        symbol="CL",
        dataset="GLBX.MDP3",
        parent_symbol="LO.OPT",
        pricing_model=PricingModel.BLACK76,
        contract_multiplier=1000,
        dividend_yield=None,
        underlying_source=UnderlyingSource.FUTURES_PRICE,
        vol_spot_multiplier=8.0,
        exercise_style=ExerciseStyle.EUROPEAN,
    ),
}


# ── Filter Thresholds ─────────────────────────────────────────

@dataclass
class FilterConfig:
    min_open_interest: int = 10
    min_bid: float = 0.01
    max_spread_ratio: float = 0.50
    moneyness_range: float = 0.20        # ±20% of underlying
    min_dte_days: float = 1.0            # minimum 1 day


FILTER_DEFAULTS = FilterConfig()


# ── Archive Configuration ─────────────────────────────────────

@dataclass
class ArchiveConfig:
    root_dir: str = "./archive"
    retention_tier1_days: int = 90       # raw chain
    retention_tier2_days: int = -1       # -1 = indefinite
    auto_purge_enabled: bool = True
    auto_purge_day: str = "sunday"
    auto_purge_hour: int = 2


ARCHIVE_DEFAULTS = ArchiveConfig(
    root_dir=os.environ.get("ARCHIVE_ROOT", "./archive")
)


# ── IV Solver Configuration ───────────────────────────────────

@dataclass
class IVSolverConfig:
    initial_guess: float = 0.20
    tolerance: float = 1e-6
    max_iterations: int = 50
    lower_bound: float = 0.001
    upper_bound: float = 5.0


IV_SOLVER_DEFAULTS = IVSolverConfig()


# ── Expiry Bucket Definitions ─────────────────────────────────

EXPIRY_BUCKETS = {
    "near_term":   (0, 2),     # 0-2 DTE
    "short_term":  (3, 7),     # 3-7 DTE
    "medium_term": (8, 30),    # 8-30 DTE
    "long_term":   (31, 9999), # 30+ DTE
}

EXPIRY_BUCKET_LABELS = {
    "near_term":   "0–2 DTE",
    "short_term":  "3–7 DTE",
    "medium_term": "8–30 DTE",
    "long_term":   "30+ DTE",
}


# ── Environment ───────────────────────────────────────────────

def get_databento_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise EnvironmentError(
            "DATABENTO_API_KEY environment variable is not set. "
            "Set it with: export DATABENTO_API_KEY=your_key_here"
        )
    return key
