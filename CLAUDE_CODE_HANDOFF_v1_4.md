# Handoff Prompt for Claude Code

**Paste the text below into Claude Code after `cd ~/greek_engine && claude`.**

The implementation brief itself lives at `IMPLEMENTATION_BRIEF_v1_4.md` in the project root — make sure it's there first (`ls IMPLEMENTATION_BRIEF_v1_4.md`).

---

## The prompt

```
Please implement v1.4 of the greek_engine project as specified in
IMPLEMENTATION_BRIEF_v1_4.md.

Before you start:

1. Run `git status` and `git log --oneline -20`. Confirm the working tree
   is clean and we're on the expected v1.3 baseline. If there are
   uncommitted changes or unexpected branches, stop and surface them
   to me — do not proceed.

2. Run `python -m pytest tests/ -q` to confirm the v1.3 baseline is green
   (143 tests passing). If anything is broken, stop and surface it.

3. Read IMPLEMENTATION_BRIEF_v1_4.md in full before writing any code.
   Also skim FUNCTIONAL_SPEC.md Section 9 (Data Lifecycle) and the
   existing pipeline.py + archive.py to understand the v1.3 patterns
   you'll be extending.

Implementation guidance:

- Implement all 7 enhancements in a single pass on a new branch
  (`git checkout -b v1.4-positioning`).
- Follow existing v1.3 code style, naming conventions, and module
  organization. Match the patterns in aggregation.py, archive.py, and
  dashboard.py.
- Write the new module (positioning.py) as a pure-functional module
  with no I/O and no globals — same shape as aggregation.py.
- The new archive directory (archive/positioning/) is independent of
  the existing two-tier Greeks archive. Do not modify the Greeks
  archive code paths.
- For Enhancement 4 (dashboard), build the new tab as an additional
  top-level tab in the existing Dash layout. Do not modify the
  existing tabs.
- Keep the chart visual extremely simple: one Plotly figure, two
  lines, standard hover, standard legend. No subplots, no fills,
  no conditional coloring, no annotations. The brief is explicit
  about this — resist the urge to add nice-to-haves.

Constraints to honor:

- DO NOT restart any systemd services. I will do that manually
  after reviewing the diff.
- DO NOT delete, archive, or migrate any existing data files.
- DO NOT modify scheduler.py — Enhancement 5 explicitly says no
  scheduler changes are needed.
- DO NOT add features marked "Non-Goals" in the brief, even if
  they seem like minor improvements (split views, alerting,
  imbalance ratio, equity products, etc.).
- DO NOT exceed the scope of the 7 enhancements as written.

Verification before you finish:

1. All 143 existing tests + 12 new tests pass: `python -m pytest tests/ -q`
2. Manual smoke test: `python engine.py positioning all` — succeeds for
   /GC and /CL, writes to archive/positioning/{GC,CL}/$(date +%Y-%m-%d)/
3. The new positioning.parquet files contain the expected columns:
   snapshot_date, snapshot_time, product, expiry, calls_total,
   puts_total, net.
4. Lint check: `python -m ruff check .` (or whatever the project uses)
5. Show me the full diff with `git diff main...HEAD` before you finish.

Deliverables when done:

- A summary of what was implemented per enhancement
- Confirmation that all tests pass (paste the pytest summary line)
- The git diff stats (`git diff --stat main...HEAD`)
- Any deviations from the brief (with rationale), or "no deviations"
- The exact commands I should run to deploy:
  - `sudo systemctl restart greek-engine`
  - Verification command for the new tab

Do not push the branch or merge to main. Leave it on the v1.4-positioning
branch for my review.
```

---

## After Claude Code finishes

Once you see the summary and the diff looks reasonable:

```bash
# Quick sanity check of what changed
git diff --stat main...v1.4-positioning

# If satisfied, merge it
git checkout main
git merge v1.4-positioning
git push  # if you push to a remote

# Restart the dashboard service to pick up the new tab
sudo systemctl restart greek-engine

# Wait for the next scheduler tick (within 30 min), then verify
# the positioning archive is populating
journalctl -u greek-scheduler -f
ls archive/positioning/GC/$(date +%Y-%m-%d)/
```

Then open `https://data.mauryinternational.com/gamma/?tab=positioning` and confirm the new tab loads with the current week's data.

---

## If anything goes wrong

Rollback is clean because nothing in v1.4 modifies existing data or services:

```bash
git checkout main                            # discard the v1.4 branch
sudo systemctl restart greek-engine          # back to v1.3 dashboard
# The archive/positioning/ directory can stay — it's ignored by v1.3 code
```

The 30-min scheduler keeps running v1.3 logic exactly as it was. No downtime, no data loss.
