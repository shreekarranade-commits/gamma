# Implementation Brief — v1.4: Positioning Tab

## For: Claude Code on droplet 192.34.63.175
## Project: ~/greek_engine
## Date: May 16, 2026

---

## Overview

This brief describes a new dashboard tab — **"Positioning"** — that displays a simplified intraday signal for futures-options products (/GC and /CL). Unlike the existing tabs, this view does **no Greek math**: no IV solve, no BSM/Black-76 evaluation, no per-contract delta or gamma. It uses a raw OI + Volume proxy as a positioning concentration signal.

All changes must maintain backward compatibility with existing archive data, pass the existing test suite (143+ tests), and add new tests for the new functionality.

---

## Conceptual Background (Read This First)

### The metric

For a selected product (/GC or /CL) and a selected expiry date, each 30-minute snapshot produces **three values**:

```
calls_total(t, expiry) = Σ_strikes_in_expiry (call_OI + call_Vol)
puts_total(t, expiry)  = Σ_strikes_in_expiry (put_OI + put_Vol)
net(t, expiry)         = calls_total − puts_total
```

`calls_total` and `puts_total` are both unsigned (always ≥ 0). `net` is signed: positive means call-heavy positioning across the chain, negative means put-heavy.

The dashboard plots only `calls_total` and `puts_total` (see Enhancement 4). `net` is stored alongside them so it's available for future use (querying, indicator overlays, alerting) without recomputation.

### Why this is different from the existing engine

- **No Greeks.** Just raw OI and Volume from the Databento `statistics` schema (already pulled by v1.3 — see Enhancement 1 of the v1.3 brief).
- **No moneyness filter.** Sum across **all strikes with non-zero OI or Volume** in the selected expiry. The user explicitly wants the full chain summed.
- **Daily expiries.** /GC and /CL options have daily expiries (Mon/Tue/Wed/Thu/Fri). The user wants to see all 5 of the current week's expiries available via dropdown.
- **OI + Volume drift is expected.** The user is aware that volume accumulates monotonically through the day and thus the metric will structurally rise during the session. They want it that way. **Do not normalize, smooth, or subtract opening volume.** Add OI and Volume raw.

### What the user said, verbatim

> "add OI + Volume and that will be my gamma...i dont want to calculate gamma anymore - just summate OI+Volume every 30 minutes and graph it on a chart for that week's expiry - I want to do this for Gold and Oil (both trade daily) - so basically I want 5 series per week and not just the friday series. i want a drop down for the 5 dates"

Treat this as the spec.

---

## Enhancement 1: Positioning Aggregation Module

### What

Compute three values — calls total, puts total, net — per (product, snapshot_date, snapshot_time, expiry).

### Where

New module: `positioning.py` at project root. Do not fold this into `aggregation.py` — keep it separate so it's clear this code path bypasses the Greeks engine entirely.

### Function signature

```python
from typing import NamedTuple

class PositioningResult(NamedTuple):
    calls_total: float
    puts_total: float
    net: float  # calls_total - puts_total

def compute_positioning(chain: pd.DataFrame, expiry: date) -> PositioningResult:
    """
    Compute calls total, puts total, and net OI+Volume for a single expiry.

    chain must have columns: strike, type ('C' or 'P'), expiry, oi, volume.
    Strikes with zero OI AND zero volume are excluded (they contribute zero
    anyway, but excluding them keeps the count honest for diagnostics).

    Returns:
        PositioningResult(calls_total, puts_total, net) summed across all
        non-zero strikes in the given expiry.
    """
```

### Implementation notes

- Filter `chain` to the requested `expiry` first
- Within the filtered slice: `calls_total = (chain[type=='C']['oi'] + chain[type=='C']['volume']).sum()`, similarly for puts
- `net = calls_total - puts_total`
- If the expiry has no contracts at all, return `PositioningResult(0.0, 0.0, 0.0)` and log a warning
- This function is **pure** — no I/O, no globals. Easy to test.

### Add a multi-expiry helper

```python
def compute_positioning_all_expiries(chain: pd.DataFrame) -> dict[date, PositioningResult]:
    """
    Returns {expiry_date: PositioningResult} for every expiry in the chain.
    """
```

Use this in the pipeline path so we compute and archive all 5 daily expiries' values from each chain pull in one pass.

---

## Enhancement 2: Pipeline Integration

### What

After the chain is built in `pipeline.py` (for /GC and /CL only — skip for SPY/QQQ/TSLA), call `compute_positioning_all_expiries(chain)` and pass the result downstream for archival.

### Where

In `pipeline.py` → `run_pipeline()`, after the chain dataframe is built and before the Greeks aggregation step. Add a block guarded by `if product.symbol in ('/GC', '/CL'):`.

### Why guarded

The user only wants this for /GC and /CL. Computing it for SPY/QQQ/TSLA is wasted work and would clutter the archive. If they extend the feature later, lifting the guard is a one-line change.

### Result shape

The pipeline should now produce, alongside its existing outputs, a positioning dict:

```python
{
    "snapshot_date": "2026-05-18",
    "snapshot_time": "10:30:00",  # 30-min tick time, market hours, ET
    "product": "/GC",
    "expiries": {
        # expiry: (calls_total, puts_total, net)
        "2026-05-18": (2_400_000.0, 1_153_000.0,  1_247_000.0),
        "2026-05-19": (  890_000.0,   972_500.0,    -82_500.0),
        "2026-05-20": (1_650_000.0, 1_240_000.0,   410_000.0),
        "2026-05-21": (  420_000.0,   405_000.0,    15_000.0),
        "2026-05-22": (1_800_000.0, 2_900_000.0, -1_100_000.0),
    }
}
```

This is the unit of archival. Each scheduler tick produces one such record per /GC and /CL.

---

## Enhancement 3: Archive Schema

### What

Persist per-snapshot positioning values in a new Parquet file separate from the existing two-tier archive.

### File location

```
archive/positioning/{product}/{snapshot_date}/positioning.parquet
```

E.g., `archive/positioning/GC/2026-05-18/positioning.parquet`.

(Use `GC` and `CL` in path names without the slash — the slash is only the display symbol.)

### Schema

| Column | Type | Notes |
|---|---|---|
| snapshot_date | date | Trading day |
| snapshot_time | time | HH:MM:SS in US/Eastern |
| product | string | "/GC" or "/CL" |
| expiry | date | The option expiry date (one of the 5 for the week) |
| calls_total | float64 | Σ (call_OI + call_Vol) across all non-zero strikes |
| puts_total | float64 | Σ (put_OI + put_Vol) across all non-zero strikes |
| net | float64 | calls_total − puts_total |

`net` is stored even though it's derivable so that reads are zero-arithmetic and the dashboard renders without recomputation.

### Write semantics

Each scheduler tick **appends** rows to the day's file. If the file doesn't exist for that date, create it. If it exists, read it, append the new rows, and write it back. Last-write-wins on duplicate (snapshot_time, expiry) — duplicates can happen if the user manually runs `engine.py run-all` mid-day. Dedupe on read by `(snapshot_time, expiry)` keeping the latest write.

### Read function

In `archive.py`, add:

```python
def load_positioning(product: str, snapshot_date: date, config) -> pd.DataFrame:
    """
    Load all intraday positioning snapshots for a single trading day.

    Returns DataFrame with columns:
        snapshot_time, expiry, calls_total, puts_total, net.
    Empty DataFrame if no archive for that date.
    """
```

---

## Enhancement 4: Dashboard Tab

### What

A new top-level tab labeled **"Positioning"** in the Plotly Dash dashboard.

### Controls (top of tab)

1. **Product selector** — radio buttons or dropdown: `/GC`, `/CL`. Default: `/GC`.

2. **Trading week selector** — dropdown of weeks (Mon–Fri groupings) that have archived positioning data. Labels: `Week of Mon 2026-05-18`. Default: the most recent week (the "current week" — defined as the Mon–Fri block containing the most recent archived trading date). A toggle labeled "Show all history" expands the dropdown to include past weeks; default is collapsed so only the current week is shown.

3. **Trading date selector** — dropdown of the 5 weekdays (Mon–Fri) within the selected week that have archived data. Days with no archive are shown but disabled (greyed out) — useful visual confirmation of which days the scheduler captured. Default: most recent archived date in the selected week.

4. **Expiry multi-select** — `dcc.Dropdown(multi=True)` populated **dynamically** with the expiries that exist in the selected trading date's archive (typically the same week's Mon–Fri, but read straight from the data — don't assume). Labels show day-of-week + date, e.g., `Mon 2026-05-18`. Default selection: the nearest expiry from the selected trading date.

5. **"Select all" button** — a small button next to the expiry dropdown that selects every expiry currently in the dropdown (will be 5 in the normal case, fewer if the archive is partial). No corresponding "Clear all" needed (the dropdown's built-in X icon on each chip handles deselection).

### URL state and deep-linking

The Positioning tab should default to the **current trading week** on load, regardless of what was last viewed. Concretely:

- `data.mauryinternational.com/gamma/?tab=positioning` → lands on current week, current trading date, nearest expiry — always, no matter when the user last visited
- `data.mauryinternational.com/gamma/?tab=positioning&date=2026-05-18` → optional, lands on the specified date if it exists in the archive; falls back to current-week behavior if not
- Internal navigation (changing dropdowns) does not need to update the URL — keep it simple

Implementation: in the Dash callback that initializes the tab, read the URL query string. If `date` is present and valid, use it. Otherwise compute "current week" from the most recent archived trading date and select that. Do **not** persist last-viewed state to localStorage or cookies — the user explicitly wants the tab to always land on "now" by default.

### Chart — Single line chart, two traces

A single Plotly line chart. No subplots, no dual panes, no sign-conditional coloring, no fills, no overlays.

**Data semantics**: at each 30-min snapshot time, the plotted value for each trace is the **sum across selected expiries**:

```
calls_plotted(t) = Σ_{expiry ∈ selected} calls_total(t, expiry)
puts_plotted(t)  = Σ_{expiry ∈ selected} puts_total(t, expiry)
```

If one expiry is selected, the chart shows that single expiry's intraday curves. If all 5 are selected, the chart shows the total OI+Volume positioning across the whole current week. Y-axis auto-rescales to fit.

**Visual spec**:
- **Type**: line chart, two traces
- **Calls trace**: line color green (`#26a69a`), label "Calls"
- **Puts trace**: line color red (`#ef5350`), label "Puts"
- **X-axis**: snapshot_time, formatted `HH:MM`, market hours only
- **Y-axis**: "OI + Volume" — both traces are unsigned positives, axis starts at 0, autoranges with selection
- **Hover**: standard Plotly hover — cursor on a point shows time and that trace's value
- **Legend**: standard Plotly legend, default placement, "Calls" and "Puts"
- **No title, no fills, no markers, no zero line, no annotations**

That's the whole spec. The visual gap between the green and red lines is the net positioning across the selected expiries — no need to plot net separately.

### Layout

Reuse the existing dashboard's color scheme and CSS. Score cards are not needed on this tab — just the controls and the single line chart. Full width. Chart height ~500px. Keep it intentionally spare.

### Empty/edge states

- No archive for selected date → display "No positioning data available for this date" in the chart area
- **Nothing selected in the expiry multi-select** → display "Select at least one expiry" in the chart area, keep controls enabled
- All selected expiries have zero rows for the day → display "No data for selected expiries"
- Only one snapshot point exists across the selected set → render single dots for each trace (don't draw one-point lines)
- A selected expiry where calls and puts are both 0 across the day (e.g., a far-out expiry that hasn't traded) → it contributes 0 to the sum; chart still renders correctly with the other selected expiries

---

## Enhancement 5: Scheduler Integration

### What

The existing scheduler (`scheduler.py`) already runs every 30 minutes during market hours and triggers a full multi-product pipeline run. **No new scheduler job is needed.** The positioning compute and archive happen inside the existing `run_pipeline()` call for /GC and /CL (gated as described in Enhancement 2), so the positioning archive populates automatically on every existing 30-min tick.

### What to verify

- The `/GC` and `/CL` legs of the existing 30-min `run_all()` call now also write to the positioning archive — confirm this in `journalctl -u greek-scheduler -f` after the next tick
- No double-writes on overlapping calls (manual `engine.py positioning` + scheduler tick at the same minute) — the last-write-wins dedupe in Enhancement 3 handles this
- Reuse the existing `is_market_open()` and holiday calendar logic — no changes needed there

---

## Enhancement 6: Engine CLI

### What

Add a CLI command for manual positioning runs and backfills:

```bash
python engine.py positioning /GC                    # one-off snapshot, archives
python engine.py positioning /GC --date 2026-05-15  # historical backfill (if Databento supports the date)
python engine.py positioning all                    # /GC and /CL
```

### Why

Useful for debugging, backfilling missed days, and manual testing without waiting for the scheduler.

---

## Enhancement 7: Retention Policy and Expired-Week Handling

### What

Positioning archives accumulate indefinitely. No automatic deletion, no compaction, no cold-storage tier. This matches the existing two-tier Greeks archive philosophy (see FUNCTIONAL_SPEC.md Section 9: Data Lifecycle Management).

### Rationale

- **OI + Volume data from expired weeks is still meaningful** — historical positioning patterns are exactly what you'd want for backtesting and pattern recognition. Deleting them would forfeit the project's most valuable asset over time.
- **Storage growth is negligible** — each day's positioning archive for /GC and /CL combined is well under 1 MB. Five years × 252 trading days × 2 products × ~50 KB/day ≈ 12 MB total. The droplet has orders of magnitude more headroom.
- **The per-day folder layout makes manual pruning trivial** — if disk pressure ever becomes real, `rm -rf archive/positioning/{product}/{old_date}/` does the job. No code needed.

### What this enhancement actually requires

Almost nothing in code:

1. **Document the retention policy** in `FUNCTIONAL_SPEC.md` under the new Positioning section (see Section 22 update at the end of this brief). State explicitly that positioning archives are kept indefinitely.
2. **No purge job, no cron task, no retention CLI.** If the user wants to delete old data manually, they can `rm -rf`.
3. **Verify the dashboard's "Show all history" toggle works** for arbitrarily-old archives, including ones predating the most recent code changes (no schema migration needed since v1.4 is the first version of the positioning archive).

### What "expired" actually means in this context (clarification)

Three different scenarios that could be called "expired" and how each is handled:

| Scenario | Handling |
|---|---|
| Option expiry dates that have already settled | Archived rows stay in place. The expiry date is just a column value; settlement doesn't trigger any action. |
| Trading days from past weeks | Archived as normal under `archive/positioning/{product}/{snapshot_date}/`. Visible in the "Show all history" view but hidden from the default current-week view. |
| Disk pressure (hypothetical, far future) | Manual `rm -rf` of old date folders. No automated process. |

None of these are deletions triggered by the system itself.

---

## Testing Requirements

Add tests to `tests/test_positioning.py`:

1. `test_compute_positioning_basic` — synthetic chain with known OI+Volume values, verify all three returned values (calls_total, puts_total, net) match hand calculation
2. `test_compute_positioning_calls_only` — chain with only calls → puts_total = 0, net = calls_total
3. `test_compute_positioning_puts_only` — chain with only puts → calls_total = 0, net = -puts_total
4. `test_compute_positioning_zero_strikes_excluded` — confirm strikes with zero OI AND zero volume don't break anything
5. `test_compute_positioning_no_matching_expiry` — chain has no contracts for requested expiry → returns PositioningResult(0, 0, 0), logs warning
6. `test_compute_positioning_all_expiries` — multi-expiry chain → returns dict with all expiries, each a PositioningResult
7. `test_load_positioning_empty_archive` — no file exists → returns empty DataFrame
8. `test_load_positioning_dedupe` — duplicate (snapshot_time, expiry) rows → only latest kept
9. `test_pipeline_skips_positioning_for_equities` — SPY pipeline run does NOT produce positioning archive
10. `test_dashboard_sums_multiple_expiries` — given a positioning DataFrame with multiple expiries, the dashboard's sum helper produces correct per-snapshot totals for any subset of selected expiries (including: single expiry, two expiries, all 5, none)
11. `test_current_week_default` — given a list of archived dates spanning multiple weeks, the "current week" helper correctly identifies the Mon–Fri block containing the most recent date
12. `test_expiry_list_dynamic_to_trading_date` — given two trading dates from different weeks, the expiry dropdown produces different (week-appropriate) options for each

Run `python -m pytest tests/ -q`. All 143 existing tests + the 12 new ones must pass.

---

## Deployment

After all changes:

1. Run tests: `python -m pytest tests/ -q`
2. Manual test: `python engine.py positioning all`
3. Verify archive: `ls archive/positioning/GC/$(date +%Y-%m-%d)/`
4. Restart the dashboard service: `sudo systemctl restart greek-engine`
5. Verify dashboard at https://data.mauryinternational.com/gamma/ — new "Positioning" tab should appear
6. Wait for the next scheduler tick (within 30 min) and verify positioning archive populates: `ls archive/positioning/GC/$(date +%Y-%m-%d)/`
7. Tail scheduler logs to confirm the tick ran cleanly: `journalctl -u greek-scheduler -f`

---

## File Change Summary

| File | Changes |
|---|---|
| positioning.py | NEW — compute_positioning, compute_positioning_all_expiries |
| pipeline.py | Add positioning compute + return path (gated to /GC, /CL) |
| archive.py | Add positioning_intraday writer + load_positioning reader |
| dashboard.py | Add Positioning tab, controls, line chart |
| engine.py | Add `positioning` CLI subcommand |
| tests/test_positioning.py | NEW — 12 tests |
| FUNCTIONAL_SPEC.md | Add Section 22 documenting the Positioning view, including the retention policy (keep indefinitely, manual purge only) |

---

## Non-Goals (Explicit)

To prevent scope creep, the following are explicitly **out of scope** for v1.4:

- Per-strike visualizations on the Positioning tab — this tab shows the *summed* metric only (across-strike totals, broken out by calls vs puts vs net)
- Greek calculations on Positioning data — this is a non-Greeks path by design
- Equity products (SPY, QQQ, TSLA) for Positioning — futures-options only for v1.4
- Backfilling historical positioning beyond what Databento's statistics schema supports
- Alerting on positioning thresholds — visual chart only for v1.4
- Imbalance ratio indicator `(calls − puts) / (calls + puts)` — interesting but deferred

Anything in this list is a future enhancement and not part of this brief.

---

*End of Implementation Brief — v1.4*
