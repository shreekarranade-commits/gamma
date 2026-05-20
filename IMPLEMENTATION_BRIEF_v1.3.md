# Implementation Brief — v1.3 Enhancements

## For: Claude Code on droplet 192.34.63.175
## Project: ~/greek_engine
## Date: April 25, 2026

---

## Overview

This brief describes 8 enhancements to implement in a single pass. The existing codebase (pipeline.py, aggregation.py, dashboard.py, archive.py, config.py, models.py, engine.py) is working and deployed at https://data.mauryinternational.com/gamma/. All changes should maintain backward compatibility with existing archive data, pass existing tests (143), and add new tests for new functionality.

---

## Enhancement 1: Capture Volume Data

### What
Pull daily trading volume per contract alongside OI from the Databento `statistics` schema. Currently we filter to `stat_type=9` (open interest) only. Volume is available as a different stat_type in the same schema.

### Why
- Enables future sign convention improvements (high volume-to-OI ratio signals directional flow vs. passive hedging)
- Flow detection: volume spikes at a strike are leading indicators for tomorrow's OI changes
- Foundation for eventual DDOI approximation

### Implementation
1. In `pipeline.py` → `build_chain_dataframe()`: query for volume stat_type alongside OI stat_type. Add a `volume` column to the chain DataFrame. If volume is unavailable, default to 0.
2. In `aggregation.py`: pass volume through to output DataFrames (no computation changes yet — volume is stored for future use).
3. In `archive.py`: ensure volume column is persisted in both Tier 1 (raw_chain.parquet) and Tier 2 (computed_greeks.parquet).
4. In `dashboard.py`: no changes yet — volume will be used in future enhancements.
5. Add `volume_to_oi_ratio` as a computed column: `volume / max(oi, 1)`.

### Databento Reference
Check `stat_type` values in the statistics schema. Volume may be stat_type=1 or another value on OPRA.PILLAR. Log available stat_types on first pull to confirm.

---

## Enhancement 2: 30-Minute Auto-Refresh During Market Hours

### What
Automatically run the pipeline for all products every 30 minutes during US market hours (9:00 AM – 5:00 PM Eastern), Monday through Friday. No polling overnight or on weekends.

### Why
Intraday exposure profiles shift as option prices and underlying prices change. 30-minute resolution captures meaningful regime changes without excessive API costs.

### Implementation
1. Create a new file `scheduler.py` that:
   - Uses `schedule` library or `APScheduler` to run every 30 minutes
   - Checks if current time is within 9:00 AM – 5:00 PM ET on a weekday
   - If yes, runs the pipeline for all products in sequence (see Enhancement 4)
   - Archives results after each run
   - Logs each run with timestamp, duration, and any errors
2. Create a systemd service file `greek-scheduler.service` that runs `scheduler.py` persistently.
3. The scheduler should handle the underlying price inference internally (see Enhancement 3).
4. Add market holiday awareness — skip US market holidays (at minimum: New Year's, MLK, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas). A simple list of dates for 2026 is sufficient.
5. Add a `--once` flag to `scheduler.py` for manual single-run testing.
6. The existing manual `engine.py run` command should continue to work independently.

### systemd Service
```ini
[Unit]
Description=Greek Engine Scheduler
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/greek_engine
Environment=DATABENTO_API_KEY=db-5CmffgJJERHE5V8aYhPtv5CgNKhhF
ExecStart=/root/greek_engine/venv/bin/python scheduler.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

---

## Enhancement 3: Put-Call Parity Spot Inference in Main Pipeline

### What
Move the spot/futures price inference logic from `test_live_spy.py` and `test_live_gc.py` into the main pipeline so that `run_pipeline()` can determine the underlying price automatically when not provided.

### Why
Currently, running the pipeline requires manually providing the underlying price. For automated 30-minute runs (Enhancement 2), the system must infer the price from the chain itself.

### Implementation
1. Create a function `infer_underlying_price(chain, snapshot_date, product, risk_free_rate)` in `pipeline.py` that:
   - For BSM products (SPY, QQQ, TSLA): uses put-call parity at ATM of nearest expiry to compute forward, then discounts to spot: `S = F * exp(-(r-q)*T)`
   - For Black-76 products (/GC, /CL): uses futures parity `F = K + (C - P) * exp(rT)` at ATM of nearest expiry
   - Falls back gracefully if no call/put pairs exist (log warning, skip product)
2. Modify `run_pipeline()` to accept `underlying_price=None` as default. When None, call `infer_underlying_price()` after building the chain.
3. The manual `engine.py run SPY 550.0` command should still accept an explicit price and use it.

---

## Enhancement 4: Multi-Product Run Command

### What
A single command that runs the full pipeline for all configured products in sequence, archives all results, and reports a summary.

### Implementation
1. Add `run_all()` function to `pipeline.py` or `engine.py` that:
   - Iterates through all products in `PRODUCTS` registry
   - For each: infers underlying price (Enhancement 3), runs pipeline, archives
   - Collects results and prints a summary table
   - Continues to next product if one fails (log error, don't abort)
2. Add CLI command: `python engine.py run-all --rate 0.05`
3. The scheduler (Enhancement 2) calls `run_all()` on each 30-minute tick.

---

## Enhancement 5: Flip Lines on Strike Profile Charts

### What
Add vertical lines on strike profile charts marking exactly where each Greek's exposure crosses zero (flips from positive to negative or vice versa).

### Why
The flip points are the most actionable information in the strike profiles:
- Gamma flip: boundary between stabilizing and destabilizing dealer hedging
- Vanna flip: the liquidity cliff — vol spike behavior changes above vs. below
- Charm flip: time decay direction reversal

### Implementation
1. In `aggregation.py`, add a function `find_flip_strikes(profiles, greek)` that:
   - Takes the strike profile DataFrame and a greek column name
   - Finds strikes where the sign changes (consecutive bars have opposite signs)
   - Interpolates between the two strikes to find the approximate zero-crossing price
   - Returns a list of flip strikes (there may be 0, 1, or multiple)
2. Store flip strikes in the output: add to scores dict as `gex_flip`, `vex_flip`, `cex_flip` (each a list of floats).
3. In `dashboard.py` → `build_strike_profile_chart()`:
   - For the GEX chart: add a vertical red dashed line at each GEX flip strike with label "GEX Flip: $XXX"
   - For the VEX chart: add a vertical red dashed line at each VEX flip with label "VEX Flip: $XXX"
   - For the CEX chart: add a vertical red dashed line at each CEX flip with label "CEX Flip: $XXX"
4. In `dashboard.py` → `build_overlay_chart()`:
   - Add all three flip lines (different colors: GEX flip in green, VEX flip in blue, CEX flip in amber) so you can see convergence/divergence
5. In the score cards section, add a small text line under each score showing the flip strike: e.g., "Flip at $548"
6. Archive the flip strikes in scores.json for historical tracking.

### Visual Style
- Line style: vertical, dashed, width=2
- GEX flip: color #ef5350 (red) on GEX chart, #26a69a (green) on overlay
- VEX flip: color #ef5350 (red) on VEX chart, #42a5f5 (blue) on overlay  
- CEX flip: color #ef5350 (red) on CEX chart, #ffa726 (amber) on overlay
- Label: positioned at top of chart, showing "Flip: $XXX.XX"

---

## Enhancement 6: Dashboard Date Comparison

### What
Ability to select two dates and view them side by side to see how positioning changed.

### Implementation
1. Add a "Compare" toggle button next to the date selector.
2. When toggled on, show a second date dropdown.
3. When two dates are selected, render the dashboard in split-view:
   - Score cards: show both dates' scores with the difference (delta) highlighted
   - Strike profiles: overlay both dates' bars (one solid, one semi-transparent) on the same chart
   - Flip lines: show both dates' flip strikes so you can see if they moved
4. Color convention: current date in normal colors, comparison date in muted/ghost colors with a legend.
5. When compare mode is off, dashboard behaves exactly as it does now (single date view).

---

## Enhancement 7: Historical Time Series Charts

### What
A new dashboard tab showing GEX, VEX, CEX, and GEX+ as line charts over time, using all archived snapshots.

### Implementation
1. Add a new tab "Time Series" to the dashboard.
2. In `archive.py`, add a function `load_score_history(product, config)` that:
   - Iterates through all archived dates for a product
   - Loads scores.json from each
   - Returns a DataFrame with columns: date, gex, vex, cex, gex_plus, underlying_price
3. In `dashboard.py`:
   - Create a time series chart with date on x-axis, score value on y-axis
   - Four lines: GEX, VEX, CEX (secondary y-axis), GEX+
   - Color-coded background bands: green when GEX+ > 0, red when GEX+ < 0
   - Hover shows exact values and underlying price on that date
   - Underlying price plotted as a secondary reference line
4. The chart should auto-scale and be filterable by date range.
5. Add flip strike history: a separate small chart showing how the GEX and VEX flip strikes moved over time relative to the underlying price. This shows whether the liquidity cliff is approaching or receding.

---

## Enhancement 8: Refresh Button Enhancement

### What
The existing Refresh button should trigger a live pipeline run (not just reload from archive) with a loading indicator.

### Implementation
1. When clicked, the dashboard calls `run_pipeline()` for the currently selected product with `underlying_price=None` (auto-inferred).
2. Show a loading spinner while the pipeline runs (~5-20 seconds).
3. Archive the new result automatically.
4. Reload the dashboard with the fresh data.
5. If the pipeline fails (e.g., Databento API error, market closed), show a toast notification with the error message instead of breaking the dashboard.
6. This is separate from the 30-minute auto-refresh (Enhancement 2). The button is for on-demand "give me the latest right now."

---

## Testing Requirements

For each enhancement, add appropriate tests:
- Enhancement 1: test that volume column exists in chain DataFrame
- Enhancement 3: test spot inference against known put-call parity values
- Enhancement 5: test flip detection with synthetic data (known zero-crossings)
- Enhancement 7: test score history loader with multiple archived dates

Run `python -m pytest tests/ -q` after all changes. All 143 existing tests must still pass, plus the new ones.

---

## Deployment

After all changes:
1. Run tests: `python -m pytest tests/ -q`
2. Restart the dashboard service: `sudo systemctl restart greek-engine`
3. Enable the scheduler service: `sudo systemctl enable greek-scheduler && sudo systemctl start greek-scheduler`
4. Verify the dashboard at https://data.mauryinternational.com/gamma/
5. Monitor first automated run in scheduler logs: `journalctl -u greek-scheduler -f`

---

## File Change Summary

| File | Changes |
|------|---------|
| pipeline.py | Add volume capture, spot inference function, run_all(), underlying_price=None default |
| aggregation.py | Add volume passthrough, flip detection function |
| dashboard.py | Flip lines on charts, compare mode, time series tab, refresh button enhancement |
| archive.py | Volume in parquet, score history loader |
| config.py | No changes expected |
| models.py | No changes expected |
| engine.py | Add run-all CLI command |
| scheduler.py | NEW — 30-min market-hours scheduler |
| greek-scheduler.service | NEW — systemd unit file |
| tests/ | New tests for volume, spot inference, flip detection, score history |
