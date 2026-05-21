# Functional Specification — Multi-Product Greek Exposure Engine

**Gamma · Vanna · Charm**

**Version 1.3 | April 2026**

*LIVING DOCUMENT — Subject to iterative revision*

---

## 1. Executive Summary

This document defines the functional specification for a Multi-Product Greek Exposure Engine that computes, aggregates, and visualizes dealer-level gamma, vanna, and charm exposures across multiple asset classes. The system ingests live and historical options chain data from Databento, derives implied volatility via Newton-Raphson inversion, computes per-contract Greeks using Black-Scholes-Merton (equities/ETFs) and Black-76 (futures) pricing models, and aggregates these into actionable exposure scores and drill-down strike profiles.

The engine is designed to support systematic identification of market regimes—stabilizing vs. destabilizing—driven by option dealer hedging dynamics. The theoretical framework draws on the SqueezeMetrics research corpus (GEX white paper, The Implied Order Book, GammaVol reference) and extends it to include charm exposure (CEX) for time-decay-driven hedging flows, multi-product support across equities and futures, and a reserved pathway for 3D surface visualization of pricing and Greek landscapes.

### 1.1 Project Goals

- Compute gamma, vanna, and charm exposure per contract, per strike, and in aggregate for each supported product.
- Produce three headline scores (GEX, VEX, CEX) in dollars-per-point units for rapid regime identification.
- Provide drill-down strike-level profiles and expiry-bucket breakdowns behind each score.
- Support multiple product classes: equity/ETF options (BSM) and futures options (Black-76).
- Enable historical analysis via date-filtered snapshots and forward-looking regime assessment.
- Reserve a pathway for 3D curved surface visualization of pricing functions and Greek landscapes (see Section 13).

### 1.2 Guiding Principles

- Correctness over speed: mathematical accuracy of Greek computations takes priority over computational performance.
- Transparency: every score must be decomposable into its per-strike, per-expiry constituents.
- Extensibility: the product configuration layer must allow new instruments to be added without modifying core logic.
- Data completeness: a single unified data pipeline must serve both the core exposure engine and any future surface visualization features without additional data sources.
- Living document: this specification will be iteratively updated as the project evolves.

---

## 2. Theoretical Foundations

### 2.1 Core Thesis

The options market is not merely a derivative of the stock market—it effectively constitutes the order book. Through mandatory delta-hedging behavior, option dealers create a measurable, predictable layer of liquidity (or illiquidity) that drives underlying price volatility more reliably than price-derived measures like VIX. This engine quantifies that layer.

Three factors cause option deltas to change, compelling dealers to re-hedge: changes in underlying price (gamma), changes in implied volatility (vanna), and changes in time (charm). By measuring dealer exposure to each of these sensitivities in dollar terms, we construct an implied order book showing where option-originated liquidity is abundant and where it is scarce.

### 2.2 Why Black-Scholes-Merton

BSM is selected not because it is the theoretically "correct" model of option pricing, but because it is the shared convention the market uses to translate between option prices and Greeks. The Implied Order Book paper describes BSM delta as "a convention"—and that word is doing significant work.

We are modeling dealer hedging behavior, not discovering option prices. The market has already set implied volatility at each strike by pricing the option. BSM, given that market-implied vol, produces the delta sensitivities that dealers actually use to determine hedge ratios. If we want to predict dealer hedging flow, we use the same function they do.

Alternative models (Heston, SABR, local vol) are valuable for pricing—fitting volatility surfaces, valuing exotics. But for computing delta sensitivities on vanilla options where implied vol is already observed at each strike, BSM with per-strike IV produces essentially identical results. The vol smile is embedded in the IV input, not assumed away.

Where BSM introduces known error is in its assumption of static IV when computing gamma—it does not account for the empirical vol-spot correlation. Rather than switching to a more complex model that internalizes this correlation, the SqueezeMetrics approach (and ours) handles it explicitly by measuring gamma and vanna as separate channels. This provides more transparency than a single model that bakes the correlation in opaquely.

### 2.3 Source Materials

| Document | Key Contribution |
|----------|-----------------|
| GEX White Paper (SqueezeMetrics, 2017) | Establishes gamma exposure framework. Demonstrates GEX quartiles predict future volatility more granularly than VIX. Defines the four assumptions for dealer positioning. |
| The Implied Order Book (SqueezeMetrics, 2020) | Introduces vanna exposure (VEX) as the crash mechanism. Shows how OTM-to-ITM moneyness transition flips vanna sign. Defines GEX+ = GEX + VEX. Introduces DDOI methodology. |
| GammaVol Reference (SqueezeMetrics) | Defines operational metrics: GIV, NPD, VGR, CR(x). Provides the 10x vol-spot multiplier convention for VEX computation. |
| Short is Long (SqueezeMetrics, 2018) | Establishes that short volume equals investor buying volume via market-maker intermediation. Provides microstructure context for dealer flow analysis. |

---

## 3. Supported Products

### 3.1 Product Registry

| Symbol | Class | Dataset | Model | Multiplier | Div Yield | Underlying |
|--------|-------|---------|-------|------------|-----------|------------|
| SPY | Equity ETF | OPRA.PILLAR | BSM | 100 | ~1.3% | Equity spot |
| QQQ | Equity ETF | OPRA.PILLAR | BSM | 100 | ~0.6% | Equity spot |
| TSLA | Equity | OPRA.PILLAR | BSM | 100 | 0.0% | Equity spot |
| /GC | Futures | GLBX.MDP3 | Black-76 | 100 | N/A | Futures price |
| /CL | Futures | GLBX.MDP3 | Black-76 | 1,000 | N/A | Futures price |

### 3.2 Product Class Differences

**Equity/ETF Options (SPY, QQQ, TSLA):** American-style exercise. Priced via Black-Scholes-Merton with continuous dividend yield adjustment. Spot price S is adjusted to S·e^(-qT) where q is the annualized dividend yield. Data sourced from Databento OPRA.PILLAR dataset.

> *American exercise note: BSM assumes European exercise. For deep ITM American-style options, early exercise premium introduces a small IV bias. Mitigated by moneyness filtering (±20%).*

**Futures Options (/GC, /CL):** European-style exercise. Priced via Black-76 where the underlying is the futures price F, not commodity spot. No dividend adjustment required. Data sourced from Databento GLBX.MDP3 dataset. Contract multipliers are product-specific.

> *Underlying distinction: for futures options, GEX measures dollars of futures buying/selling per point move in the futures price, not per dollar move in commodity spot.*

### 3.3 Product Configuration Schema

| Field | Type | Description |
|-------|------|-------------|
| symbol | String | Trading symbol identifier |
| dataset | String | Databento dataset (OPRA.PILLAR or GLBX.MDP3) |
| parent_symbol | String | Databento parent symbol for chain pull (e.g., SPY.OPT) |
| pricing_model | Enum | BSM or BLACK76 |
| contract_multiplier | Integer | Shares/units per contract |
| dividend_yield | Float or None | Annualized continuous dividend yield |
| underlying_source | Enum | EQUITY_SPOT or FUTURES_PRICE |
| vol_spot_multiplier | Float | Anti-correlated vol-spot ratio for VEX computation |
| exercise_style | Enum | AMERICAN or EUROPEAN |

---

## 4. Data Architecture

### 4.1 Primary Data Source: Databento

All market data is sourced from Databento via their Python SDK. The subscriber account provides access to both OPRA.PILLAR (US equity options) and GLBX.MDP3 (CME futures options).

### 4.2 Required Schemas Per Product

| Schema | Data Elements | Usage | Frequency |
|--------|--------------|-------|-----------|
| definition | Strike, expiry, option type, symbol, instrument class | Chain skeleton; maps instrument_id to contract attributes | Daily / on-demand |
| mbp-1 / cmbp-1 | Best bid/ask per contract | Mid price computation for IV solver input | Snapshot near close or intraday |
| statistics | Open interest AND volume per contract | OI: weighting factor for exposure. Volume: flow detection, sign convention improvement | End-of-day (OI) / intraday (volume) |

### 4.3 Supplementary Data Sources

| Data Element | Source | Method | Update Frequency |
|-------------|--------|--------|-----------------|
| Underlying spot price (equities) | Databento equities feed or derived | API pull or last trade | Intraday / per snapshot |
| Underlying futures price | Databento GLBX.MDP3 (front month) | API pull or last trade | Intraday / per snapshot |
| Risk-free rate | US Treasury yield curve or Fed Funds | External API or manual input | Daily |
| Dividend yield (SPY, QQQ) | Market data or manual estimate | Annualized continuous yield | Quarterly review |

> *Risk-free rate impact: the papers set r = 0 (adequate during ZIRP era). With current rate levels, r appears in d1, in the BSM/Black-76 pricing formula for IV solving, and explicitly in the charm formula. For options beyond 30 DTE, the rate shift on d1 is material.*

### 4.4 Data Sufficiency for Future Features

The data pipeline is sufficient for all planned features, including the reserved 3D surface visualization (Section 13). No additional data sources or API calls are needed to support surface features.

---

## 5. Data Quality and Filtering

### 5.1 Rationale

Raw options chain data contains contracts that are illiquid, mispriced, or too far from the money to contribute meaningful exposure. Filtering is applied before IV computation.

### 5.2 Filter Rules

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Minimum open interest | OI >= 10 | Negligible OI contributes no meaningful exposure |
| Valid bid required | Bid > 0 | Zero-bid contracts have no market; IV solver will fail |
| Maximum bid-ask spread | (Ask - Bid) / Mid <= 0.50 | Wide spreads produce unreliable IV estimates |
| Moneyness window | Strike within ±20% of underlying | Deep OTM/ITM options contribute minimal exposure |
| Minimum time to expiry | T >= 1/365 (at least 1 day) | Sub-day expiries cause division-by-near-zero in formulas |

### 5.3 IV Solver Failure Handling

Contracts where the Newton-Raphson solver does not converge within 50 iterations are flagged and excluded. The count is logged per snapshot.

---

## 6. Mathematical Framework

### 6.1 Common Foundations

All Greek computations flow through the standard normal distribution:

- **d1** = [ln(S/K) + (r - q + σ²/2)·T] / (σ√T)
- **d2** = d1 - σ√T
- **φ(d1)** = (1/√2π) · e^(-d1²/2) — standard normal PDF
- **N(x)** = cumulative distribution function of the standard normal

Note: all three Greeks share φ(d1) in the numerator. Both vanna and charm contain d2, making them moneyness-dependent.

### 6.2 Model A: Black-Scholes-Merton (Equities/ETFs)

**Pricing Functions (for IV Solver):**
- Call Price = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
- Put Price = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
- Vega = S·e^(-qT)·φ(d1)·√T

**Delta:**
- Call Delta = e^(-qT)·N(d1)
- Put Delta = -e^(-qT)·N(-d1)

**Gamma (∂Δ/∂S):**
- Gamma = e^(-qT)·φ(d1) / (S·σ·√T) — identical for calls and puts, always positive

**Vanna (∂Δ/∂σ):**
- Vanna = -e^(-qT)·φ(d1)·d2 / σ — sign depends on moneyness via d2

**Charm (∂Δ/∂T):**
- Call Charm = -q·e^(-qT)·N(d1) + e^(-qT)·φ(d1)·[2(r-q)T - d2·σ√T] / (2T·σ√T)
- Put Charm = q·e^(-qT)·N(-d1) + e^(-qT)·φ(d1)·[2(r-q)T - d2·σ√T] / (2T·σ√T)

### 6.3 Model B: Black-76 (Futures Options)

**d1** = [ln(F/K) + (σ²/2)·T] / (σ√T) — note: r does not appear in d1

**Pricing:**
- Call = e^(-rT)·[F·N(d1) - K·N(d2)]
- Put = e^(-rT)·[K·N(-d2) - F·N(-d1)]

**Greeks:**
- Gamma = e^(-rT)·φ(d1) / (F·σ·√T)
- Vanna = -e^(-rT)·φ(d1)·d2 / σ
- Charm: Call = -r·e^(-rT)·N(d1) - e^(-rT)·φ(d1)·d2/(2T); Put = r·e^(-rT)·N(-d1) - e^(-rT)·φ(d1)·d2/(2T)

### 6.4 Implied Volatility Solver

Databento does not provide pre-calculated IV. IV is derived via Newton-Raphson iteration:

1. Initialize: σ₀ = 0.20
2. Compute model price P(σₙ)
3. Compute vega V(σₙ)
4. Update: σₙ₊₁ = σₙ - [P(σₙ) - P_market] / V(σₙ)
5. Convergence: |σₙ₊₁ - σₙ| < 10⁻⁶, max 50 iterations
6. Bounds: clamp σ to [0.001, 5.0]

Edge cases: vega near zero (deep ITM/OTM) causes instability — contracts are flagged and excluded. Negative mid prices are excluded. Very high IV (>300%) is accepted but logged.

---

## 7. Exposure Aggregation

### 7.1 Sign Convention (Approach A)

Without DDOI transaction data, dealer positioning is inferred:
- **Calls:** customers sell → dealers long → sign = +1
- **Puts:** customers buy → dealers short → sign = -1

### 7.2 GEX (Gamma Exposure)

**GEX** = Σ [Gamma_i × OI_i × Multiplier × sign_i × S] — dollars per underlying point

Positive GEX: dealers buy dips and sell rips (stabilizing). Negative GEX: dealers sell into drops (destabilizing).

### 7.3 VEX (Vanna Exposure)

**VEX** = Σ [Vanna_i × Δσ_per_point × OI_i × Multiplier × sign_i × S]

Where Δσ_per_point = σ_i × VolMultiplier / S. Same units as GEX (dollars per point).

### 7.4 CEX (Charm Exposure)

**CEX** = Σ [-Charm_i × OI_i × Multiplier × sign_i × S] / 365 — dollars per day

Negated because charm = dΔ/dT but time passing means T decreases. Unlike GEX and VEX, CEX is unconditional — time passes regardless.

### 7.5 Vol-Spot Multiplier Configuration

| Product | Default Multiplier | Rationale |
|---------|-------------------|-----------|
| SPY | 10x | Well-calibrated historical relationship from SqueezeMetrics |
| QQQ | 10x | Similar index-level dynamics |
| TSLA | 15x (configurable) | Higher beta; larger IV reactions |
| /GC | 5x (configurable) | Weaker vol-spot relationship; sometimes positive |
| /CL | 8x (configurable) | Varies with macro regime |

### 7.6 GEX+ (Composite Score)

**GEX+** = GEX + VEX — total option-originated top-of-book liquidity in dollars per point.

---

## 8. Computation Pipeline

For each product, per snapshot date:

1. **Product Config:** Load product configuration entry.
2. **Ingest Definitions:** Pull instrument definitions from Databento.
3. **Ingest Quotes:** Pull top-of-book snapshot.
4. **Ingest Statistics:** Pull end-of-day open interest.
5. **Join:** Merge on instrument_id to produce one row per contract.
6. **Underlying Price:** Retrieve spot or futures price.
7. **Reference Data:** Load risk-free rate and dividend yield.
8. **Filter:** Apply data quality filters (Section 5.2).
9. **IV Solve:** Vectorized Newton-Raphson (Section 6.4).
10. **Greek Compute:** Calculate gamma, vanna, charm per contract.
11. **Sign & Scale:** Apply sign convention, scale to dollar exposure.
12. **Aggregate:** Sum for headline scores; retain per-strike for profiles.
13. **Profile:** Retain per-strike-per-expiry for breakdown view.
14. **Archive:** Persist to archive store (Section 9).

---

## 9. Data Lifecycle Management

### 9.1 Design Rationale

Historical archives enable backtesting, regime analysis, and model calibration. Without them, only current regime assessment is possible.

### 9.2 Two-Tier Storage Architecture

| Tier | Contents | Format | Purpose |
|------|----------|--------|---------|
| Tier 1: Raw Chain | Per-contract row: strike, expiry, type, bid, ask, mid price, OI | Parquet | Full replay capability |
| Tier 2: Computed Results | Scores, per-strike profiles, per-contract Greeks, metadata | Parquet + JSON | Fast retrieval without recomputation |

### 9.3 Directory Structure

```
archive/
  SPY/
    2026-04-22/
      raw_chain.parquet
      computed_greeks.parquet
      strike_profiles.parquet
      expiry_breakdown.parquet
      scores.json
      metadata.json
    2026-04-23/
      ...
  QQQ/
    ...
```

### 9.4 Retention Policies

| Tier | Default Retention | Configurable |
|------|------------------|-------------|
| Tier 1: Raw Chain | 90 days | RETENTION_TIER1_DAYS |
| Tier 2: Computed Results | Indefinite | RETENTION_TIER2_DAYS (default: -1) |

### 9.5 Purge Process

Automatic weekly purge (Sunday 02:00) plus manual CLI:

```bash
python engine.py purge --product SPY --before 2026-01-01 --tier 1
python engine.py purge --product ALL --before 2025-12-31 --tier all --dry-run
```

### 9.6 Replay Capability

Archived Tier 1 snapshots can be replayed with different parameters without consuming Databento API credits.

### 9.7 Storage Estimates

All 5 products: ~6.2 MB/day raw, ~330 KB/day computed. ~588 MB per quarter at full retention.

---

## 10. Output Specification

### 10.1 Headline Scores

| Score | Units | Interpretation |
|-------|-------|---------------|
| GEX | $ per underlying point | Net dealer gamma hedging flow. Positive = stabilizing. |
| VEX | $ per underlying point | Net dealer vanna hedging flow. Negative = crash risk. |
| CEX | $ per day | Net dealer charm-driven adjustment. Dominates near OPEX. |
| GEX+ | $ per underlying point | GEX + VEX. Composite liquidity. |

### 10.2 Strike-Level Profiles

Per-strike exposure bars showing gamma walls, vanna flip points, and charm cliffs.

### 10.3 Expiry-Bucket Breakdown

| Bucket | Range | Dominant Greek |
|--------|-------|---------------|
| Near-term | 0–2 DTE | Charm |
| Short-term | 3–7 DTE | Gamma |
| Medium-term | 8–30 DTE | Balanced |
| Long-term | 30+ DTE | Vanna |

### 10.4 Combined Overlay View

All three Greeks overlaid on a single strike axis with dual y-axes.

### 10.5 Per-Contract Detail (CSV Export)

Full per-contract dataset available for debugging and advanced analysis.

---

## 11. Visualization Specification

### 11.1 Design Philosophy

Score-first, detail-on-demand hierarchy.

### 11.2 View Hierarchy

- **Layer 1 — Headline Dashboard:** Four score cards with color coding. Product selector.
- **Layer 2 — Strike Profiles:** Bar charts per Greek with spot reference line.
- **Layer 3 — Expiry Breakdown:** Stacked bars by expiry bucket.
- **Layer 4 — Combined Overlay:** All three Greeks on one chart.

### 11.3 Controls and Filters

- Product selector (SPY, QQQ, TSLA, /GC, /CL)
- Date selector (historical snapshots from archive)
- Expiry filter (toggle buckets)
- Moneyness zoom (±5% to ±20%)
- Normalization toggle

### 11.4 Technology

Plotly Dash (Python-native). Served via gunicorn behind nginx with SSL at https://data.mauryinternational.com/gamma/.

---

## 12. Numerical Validation

### 12.1 Reference Case: OTM Put

Setup: S=3000, K=2900, T=30/365, σ=0.20, r=0, q=0

| Parameter | Value |
|-----------|-------|
| d1 | 0.6199 |
| d2 | 0.5626 |
| φ(d1) | 0.3292 |
| Put Delta (unsigned) | 0.2676 |
| Gamma | 0.001914 |
| Charm (daily) | 0.003087 |

### 12.2 Vanna Sign Flip Verification

| Parameter | OTM (S=3000) | ITM (S=2800) |
|-----------|-------------|-------------|
| d2 | +0.5626 | -0.6410 |
| Vanna direction | Vol up → dealer buys | Vol up → dealer sells |

---

## 13. Reserved Feature: 3D Surface Visualization

Candidate surfaces: volatility surface, option price surface, Greek surfaces, scenario P&L surface. Requires no additional data. Implementation deferred.

---

## 14. Technical Requirements

### 14.1 Language and Environment

- Core: Python 3.10+ with NumPy, SciPy
- Data: Databento Python SDK
- Visualization: Plotly Dash
- Archive: Parquet (pyarrow) + JSON

### 14.2 Performance Requirements

- Single product chain computation: < 5 seconds
- All five products sequentially: < 30 seconds
- IV solver: tolerance 10⁻⁶, max 50 iterations

### 14.3 Accuracy Requirements

- Greeks match manual calculation to within 10⁻⁴ relative error for ATM options
- Reference case (Section 12.1) must reproduce exactly
- IV solver failures logged and excluded

### 14.4 Build Prerequisites

1. Databento API key (environment variable DATABENTO_API_KEY)
2. Visualization: Plotly Dash ✅
3. Starting product: SPY ✅

---

## 15. Automated Scheduling (v1.3)

### 15.1 30-Minute Intraday Refresh

The pipeline runs automatically every 30 minutes during US market hours (9:00 AM – 5:00 PM Eastern), Monday through Friday. No polling overnight, weekends, or US market holidays.

Each tick executes a multi-product run (Section 15.3) that pulls fresh option chain data from Databento, infers underlying prices, computes Greeks, aggregates exposures, and archives results. The scheduler runs as a persistent systemd service on the droplet.

Holiday calendar covers: New Year's, MLK Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas (2026 dates hardcoded, extensible).

### 15.2 Underlying Price Inference

The pipeline infers the underlying price automatically via put-call parity from the options chain itself, eliminating the need for a separate equity data feed or manual price input.

For BSM products (SPY, QQQ, TSLA): Find the ATM strike K* on the nearest expiry where |C - P| is minimized. Compute forward: F = K* + C(K*) - P(K*). Discount to spot: S = F · exp(-(r-q)·T).

For Black-76 products (/GC, /CL): Compute futures price directly: F = K* + (C - P) · exp(r·T) at ATM of nearest expiry.

The `run_pipeline()` function accepts `underlying_price=None` as default and infers automatically. Explicit prices are still accepted for manual runs.

### 15.3 Multi-Product Run

A single `run_all()` function and CLI command (`python engine.py run-all`) executes the pipeline for all configured products in sequence: SPY → QQQ → TSLA → /GC → /CL. Each product's results are archived independently. If one product fails, the error is logged and the run continues to the next product.

## 16. Volume Data Capture (v1.3)

### 16.1 Rationale

Daily trading volume per contract is captured alongside open interest. Volume enables three future capabilities:

- **Sign convention improvement:** High volume-to-OI ratio at a strike signals active directional positioning rather than passive hedging. This allows per-strike sign refinement beyond the blanket Approach A assumption.
- **Flow detection:** Volume spikes at previously low-OI strikes are leading indicators for positioning changes. Today's high-volume strikes become tomorrow's GEX-moving OI.
- **DDOI foundation:** Volume is the prerequisite for inferring trade direction, which is the path toward Dealer Directional Open Interest without transaction-level feeds.

### 16.2 Implementation

Volume is pulled from the Databento `statistics` schema alongside OI (different stat_type value). A `volume` column and `volume_to_oi_ratio` computed column are added to the chain DataFrame. Both are persisted in Tier 1 and Tier 2 archive files. No computation changes in v1.3 — volume is stored for future use and display.

## 17. Flip Line Detection (v1.3)

### 17.1 Definition

A "flip" is the strike price where a Greek's aggregate exposure crosses zero — switching from positive (stabilizing/providing) to negative (destabilizing/taking) or vice versa.

- **Gamma flip:** The strike where GEX changes sign. Below this price, dealer gamma hedging destabilizes; above it, it stabilizes (or vice versa).
- **Vanna flip:** The strike where VEX changes sign. This is the liquidity cliff identified in The Implied Order Book. Above it, vol spikes cause dealer buying. Below it, vol spikes cause dealer selling.
- **Charm flip:** The strike where CEX changes sign. Marks the reversal point for time-decay-driven hedging direction.

### 17.2 Detection Algorithm

For each Greek's strike profile, scan consecutive strikes for sign changes. When the exposure at strike K_n and K_{n+1} have opposite signs, interpolate linearly to find the zero-crossing:

Flip strike = K_n + (K_{n+1} - K_n) · |exposure(K_n)| / (|exposure(K_n)| + |exposure(K_{n+1})|)

There may be 0, 1, or multiple flips per Greek. All are reported.

### 17.3 Visualization

On each strike profile chart, flip points are rendered as vertical dashed lines with labels showing the exact strike price. On the combined overlay chart, all three flips appear together in distinct colors (GEX: green, VEX: blue, CEX: amber) to show convergence or divergence.

When flip lines converge at a similar strike, it indicates a fragility hotspot where all three dealer dynamics reverse simultaneously.

Flip strikes are stored in scores.json for historical tracking.

### 17.4 Score Card Integration

Each score card displays the flip strike(s) beneath the headline number: e.g., "Flip at $548.30".

## 18. Dashboard Enhancements (v1.3)

### 18.1 Date Comparison Mode

A "Compare" toggle and second date dropdown allow side-by-side analysis of two snapshots:

- Score cards show both dates' values with the delta highlighted (e.g., GEX: $88M → $72M, Δ: -$16M)
- Strike profile charts overlay both dates' bars (current date solid, comparison date semi-transparent)
- Flip lines from both dates shown so movement of flip points is visible
- Legend distinguishes current vs. comparison date

### 18.2 Historical Time Series Tab

A new "Time Series" tab plots GEX, VEX, CEX, and GEX+ as line charts over all archived dates:

- Date on x-axis, exposure value on y-axis
- Four lines: GEX (green), VEX (blue), CEX (amber, secondary y-axis), GEX+ (white)
- Background shading: green band when GEX+ > 0, red band when GEX+ < 0
- Underlying price as a secondary reference line
- Hover shows exact values and underlying price
- Date range filter
- Separate sub-chart showing flip strike movement over time relative to underlying price

### 18.3 Live Refresh Button

The Refresh button triggers a live pipeline run for the currently selected product (not just a reload from archive):

- Calls `run_pipeline()` with auto-inferred underlying price
- Shows loading spinner during execution (~5-20 seconds)
- Archives the new result automatically
- Reloads dashboard with fresh data
- On failure (API error, market closed), shows toast notification without breaking the dashboard

### 18.4 Regime-Aware Score Card Interpretation (v1.3)

Individual score cards should not interpret themselves in isolation. The real signal is in the combination of all three scores. Each card's subtitle reflects the cross-score regime context.

#### 18.4.1 Overall Regime Banner

A colored banner displayed above the four score cards showing the one-glance regime classification:

| Banner | Color | Condition |
|--------|-------|-----------|
| STABILIZING | Green | GEX > 0 and VEX > 0 |
| FRAGILE | Amber | GEX > 0 and VEX < 0 |
| DESTABILIZED | Red | GEX < 0 and VEX < 0 |
| NEUTRAL | Gray | GEX and VEX both near zero |

#### 18.4.2 GEX Card Subtitles

| GEX | VEX | Subtitle |
|-----|-----|----------|
| Positive | Positive | "Stabilizing · Dealers absorb moves" |
| Positive | Negative | "Stable but fragile · Vol spike risk" |
| Negative | Negative | "Destabilized · Moves amplified" |
| Negative | Positive | "Weak · Gamma negative, vanna supportive" |
| Near zero | Any | "Neutral · No gamma pressure" |

#### 18.4.3 VEX Card Subtitles

| VEX | GEX | Subtitle |
|-----|-----|----------|
| Positive | Positive | "Vol spike = dealer buying · Protected" |
| Negative | Positive | "Vol spike = dealer selling · Liquidity cliff below" |
| Negative | Negative | "Crash-prone · Both channels destabilize" |
| Positive | Negative | "Vanna cushion · Partial protection" |

#### 18.4.4 CEX Card Subtitles

| Condition | Subtitle |
|-----------|----------|
| CEX > 0 (large) | "Sellers tomorrow · Downward drift into OPEX" |
| CEX < 0 (large) | "Buyers tomorrow · Upward drift into OPEX" |
| CEX near zero | "Minimal time decay pressure" |
| Nearest expiry within 2 DTE | Prepend "OPEX: " to any label above |

#### 18.4.5 GEX+ Card Subtitles

| Condition | Subtitle |
|-----------|----------|
| GEX+ > 0 (large) | "Strong liquidity · Range-bound likely" |
| GEX+ > 0 (small) | "Mild support · Normal conditions" |
| GEX+ < 0 | "Liquidity vacuum · Trend/crash risk" |
| GEX+ near zero | "Balanced · Options not driving price" |

#### 18.4.6 Threshold Definitions

"Near zero" for regime classification is defined as within 10% of the product's recent historical range (rolling 20-day percentile). Until sufficient history is available, a fixed threshold is used: |GEX| < $5M for SPY, scaled proportionally for other products.

"Large" for CEX and GEX+ is defined as above the 75th percentile of recent history. Until sufficient history: |CEX| > $50M for SPY, |GEX+| > $100M for SPY.

Flip strike text remains below the subtitle as currently implemented.

## 19. Historical Backfill (v1.3)

### 19.1 Purpose

A single snapshot tells you the current regime. Historical analysis requires archived data across many trading days. The backfill capability replays the pipeline against historical dates using Databento's historical data API, building a retroactive archive that enables time series analysis, regime pattern recognition, flip line tracking, and framework validation.

### 19.2 Use Cases

- **Pre-move exposure analysis:** Examine what GEX/VEX/CEX looked like in the days before significant market moves. Identify whether negative VEX consistently precedes selloffs, validating the crash mechanism thesis.
- **Score contextualization:** Establish normal ranges for each score per product. A GEX reading is only meaningful relative to its historical distribution — is $88M high, low, or average for SPY?
- **Flip line drift tracking:** Monitor how gamma and vanna flip strikes move over time relative to the underlying price. A flip line approaching the current price signals shrinking safety margin even if headline scores haven't changed.
- **Framework validation:** Correlate historical GEX levels with subsequent realized volatility. Test whether the SqueezeMetrics thesis (high GEX → low realized vol, negative VEX → crash) holds on your own data and products.
- **Regime transition study:** Overlay GEX+ time series with underlying price to observe the cycle: high GEX+ → tight range → declining GEX+ → expanding range → negative GEX+ → large directional moves.

### 19.3 Backfill Script

A standalone `backfill.py` script and CLI command (`python engine.py backfill`) that:

1. Accepts parameters: product symbol, start date, end date (default: today), risk-free rate, delay between API calls
2. Generates a list of valid trading days (excluding weekends and US market holidays)
3. For each trading day: runs the full pipeline with auto-inferred underlying price (Section 15.2), 15:45 ET snapshot time, and archives the result
4. Enforces a configurable delay between days (default: 2 seconds) to avoid Databento API rate limiting
5. Logs progress per day: date, headline scores, computation time
6. On per-day failure: logs the error and continues to the next day without aborting
7. Prints a summary on completion: total days processed, successes, failures, total elapsed time
8. Supports `--dry-run` flag to list target dates without making API calls

### 19.4 CLI Interface

```
python engine.py backfill --product SPY --start 2026-01-02 --end 2026-04-25 --delay 2
python engine.py backfill --product SPY --start 2026-01-02 --dry-run
python engine.py backfill --product QQQ --start 2026-03-01 --delay 3
```

### 19.5 API Cost Considerations

Each backfill day consumes three Databento API calls (definitions, quotes, statistics) per product. A full 80-day backfill across 5 products = ~1,200 API calls. The recommended approach is to backfill SPY first, verify data consumption, then extend to other products based on plan capacity.

Backfilled data is stored in the same archive structure as live data (Section 9) and is immediately available to the time series tab (Section 18.2), date comparison mode (Section 18.1), and all other dashboard views.

### 19.6 Recommended Backfill Strategy

| Phase | Product | Start Date | Estimated Days | API Calls |
|-------|---------|-----------|----------------|-----------|
| 1 | SPY | 2026-01-02 | ~80 | ~240 |
| 2 | QQQ | 2026-01-02 | ~80 | ~240 |
| 3 | TSLA | 2026-01-02 | ~80 | ~240 |
| 4 | /GC | 2026-01-02 | ~80 | ~240 |
| 5 | /CL | 2026-01-02 | ~80 | ~240 |

Run phases sequentially, monitoring API usage between each. Phase 1 alone provides sufficient history for meaningful SPY time series analysis.

## 20. Known Limitations and Assumptions

1. **No DDOI data:** dealer positioning inferred from standard assumptions.
2. **American exercise approximation:** BSM assumes European. Mitigated by moneyness filtering.
3. **Static vol-spot multiplier:** fixed per product, not dynamically calibrated.
4. **Snapshot-based:** not real-time streaming.
5. **Flat risk-free rate:** single rate across all expiries.
6. **Continuous dividend yield:** not discrete payments.
7. **Charm magnitude:** smallest of three Greeks in dollar impact; most valuable near OPEX.

---

## 21. Future Enhancements

- DDOI integration (transaction-level dealer positioning)
- Real-time streaming via Databento Live API
- Composite Fragility Score (GEX + VEX + CEX + CR(x))
- Dynamic vol-spot multiplier calibration
- 3D surface visualization (Section 13)
- Multi-product cross-exposure analysis
- Alerting on regime transitions
- Bjerksund-Stensland upgrade for American options
- Term structure interpolation for risk-free rate
- Volume/OI heuristics for improved sign convention

---

## 22. Positioning View (v1.4)

A simplified, non-Greeks intraday signal for futures-options products
(/GC and /CL). Bypasses the IV solve, BSM/Black-76 evaluation, and per-
contract Greek computation that drive the rest of the engine.

### 22.1 Metric

For a selected product and expiry date, each 30-minute snapshot produces
three values summed across every strike with non-zero OI or volume:

```
calls_total(t, expiry) = Σ_strikes (call_OI + call_Vol)
puts_total(t, expiry)  = Σ_strikes (put_OI + put_Vol)
net(t, expiry)         = calls_total − puts_total
```

`calls_total` and `puts_total` are always ≥ 0. `net` is signed and stored
alongside them so the dashboard renders without recomputation and future
queries (overlays, alerting) need no arithmetic.

No moneyness filter is applied — the full chain is summed. OI + Volume
drift is expected (volume accumulates monotonically during the session);
the metric is intentionally not normalized or detrended.

### 22.2 Pipeline Integration

Positioning is computed inside `run_pipeline()` for /GC and /CL only,
immediately after `build_chain_dataframe()` and before any filtering or
IV solve. SPY/QQQ/TSLA skip the path silently. Each scheduler tick (30
minutes during market hours) emits one record per /GC and /CL with all
of that week's daily expiries.

### 22.3 Archive Schema

Per-trading-day Parquet files independent of the Greeks two-tier archive:

```
archive/positioning/{GC|CL}/{snapshot_date}/positioning.parquet
```

Columns:

| Column | Type | Notes |
|---|---|---|
| snapshot_date | date | Trading day |
| snapshot_time | string | HH:MM:SS in US/Eastern, seconds zeroed |
| product | string | "/GC" or "/CL" |
| expiry | date | One of the week's daily expiries |
| calls_total | float64 | Σ (call_OI + call_Vol) |
| puts_total | float64 | Σ (put_OI + put_Vol) |
| net | float64 | calls_total − puts_total |

Each tick appends rows. Last-write-wins on duplicate `(snapshot_time,
expiry)` keys — manual CLI overlap with the scheduler is handled by
dedupe on read and write.

### 22.4 Dashboard Tab

A "Positioning" tab in the main Dash view-tabs row. Controls:

- Product radio (/GC, /CL — default /GC)
- "Show all history" switch (default off → current week only)
- Trading-week dropdown (Mon–Fri label)
- Trading-date dropdown (5 weekdays of the chosen week; days without
  archived data shown but disabled)
- Multi-select expiry dropdown (populated dynamically from the day's
  archive; default selects the nearest expiry)
- "Select all" button (selects every enabled expiry)

The chart is a single Plotly line chart with two traces — Calls
(`#26a69a`) and Puts (`#ef5350`). X-axis is `snapshot_time` formatted as
HH:MM; Y-axis is "OI + Volume" anchored at 0. No fills, no markers
(unless only one snapshot point exists), no zero line, no subplots, no
sign-conditional coloring. The vertical gap between the two lines is the
net positioning across the selected expiries.

URL deep-linking: `?tab=positioning` lands on the Positioning tab,
`?tab=positioning&date=YYYY-MM-DD` additionally prefills that trading
date when archived. The tab always defaults to "now" (current week,
most recent date, nearest expiry) on fresh loads; no per-user state is
persisted.

### 22.5 Retention Policy

**Positioning archives are kept indefinitely.** No automatic purge, no
cold-storage tier, no compaction.

Rationale:
- Historical OI + Volume from expired weeks is the project's most
  valuable asset for backtesting and pattern recognition.
- Storage growth is negligible — ~50 KB/day × 2 products × 252 days/yr
  ≈ 25 MB/year.
- The per-day folder layout makes manual pruning trivial:
  `rm -rf archive/positioning/{product}/{date}/`. No code required.

Three "expired" scenarios and how each is handled:

| Scenario | Handling |
|---|---|
| Option expiry dates that have settled | Archived rows stay; the expiry is just a column value. |
| Past trading days | Archived normally; visible via "Show all history". |
| Disk pressure (hypothetical) | Manual `rm -rf`. No automated process. |

### 22.6 CLI

```bash
python engine.py positioning /GC
python engine.py positioning /GC --date 2026-05-15
python engine.py positioning all
```

Useful for backfills and manual testing without waiting for the
scheduler. Manual runs share the same dedupe semantics as scheduler
runs.

---

## 23. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 22, 2026 | Initial functional specification. |
| 1.1 | April 23, 2026 | Major revision: theoretical foundations, product config schema, expanded math framework, numerical validation, 3D surface reserved feature, build prerequisites, future enhancements. |
| 1.2 | April 23, 2026 | Added Data Lifecycle Management (Section 9): two-tier archive, Parquet storage, retention policies, purge processes, replay capability. Added archive step to pipeline. |
| 1.3 | April 25, 2026 | Added: 30-minute automated scheduling during market hours (Section 15). Put-call parity spot inference in main pipeline (Section 15.2). Multi-product run command (Section 15.3). Volume data capture alongside OI (Section 16). Flip line detection and visualization for gamma, vanna, and charm zero-crossings (Section 17). Dashboard date comparison mode (Section 18.1). Historical time series tab (Section 18.2). Live refresh button enhancement (Section 18.3). Regime-aware score card interpretation with cross-score context and overall regime banner (Section 18.4). Historical backfill capability with CLI interface and phased rollout strategy (Section 19). Renumbered Sections 20–22. |
| 1.4 | May 16, 2026 | Added Positioning view (Section 22): non-Greeks intraday OI+Volume signal for /GC and /CL, separate per-day Parquet archive kept indefinitely, dedicated dashboard tab with current-week default and "Show all history" toggle, `positioning` CLI subcommand. Renumbered Revision History to Section 23. |

---

*End of Functional Specification — Living Document — v1.4*
