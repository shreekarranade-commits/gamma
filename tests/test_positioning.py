"""
Tests for v1.4 positioning: compute, archive, pipeline gate, and
dashboard helpers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
import pytest

from positioning import (
    PositioningResult, compute_positioning, compute_positioning_all_expiries,
    current_week_dates, sum_positioning_across_expiries,
)
from archive import (
    archive_positioning, load_positioning, list_positioning_dates,
    get_positioning_path,
)
from config import ArchiveConfig, PRODUCTS
from pipeline import maybe_archive_positioning


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_archive(tmp_path):
    return ArchiveConfig(root_dir=str(tmp_path / "archive"))


@pytest.fixture
def two_expiry_chain():
    """
    Synthetic chain across two expiries with hand-computed totals.

    Expiry 2026-05-22: calls oi+vol = (100+10)+(50+5)=165, puts = (80+8)+(40+4)=132
    Expiry 2026-05-29: calls oi+vol = (200+20)+(100+10)=330, puts = (160+16)=176
    """
    return pd.DataFrame([
        # Expiry 1 — calls
        {"strike": 2000.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 100, "volume": 10},
        {"strike": 2010.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 50, "volume": 5},
        # Expiry 1 — puts
        {"strike": 2000.0, "type": "P", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 80, "volume": 8},
        {"strike": 2010.0, "type": "P", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 40, "volume": 4},
        # Expiry 2 — calls
        {"strike": 2000.0, "type": "C", "expiry": pd.Timestamp("2026-05-29"),
         "oi": 200, "volume": 20},
        {"strike": 2020.0, "type": "C", "expiry": pd.Timestamp("2026-05-29"),
         "oi": 100, "volume": 10},
        # Expiry 2 — puts
        {"strike": 2000.0, "type": "P", "expiry": pd.Timestamp("2026-05-29"),
         "oi": 160, "volume": 16},
    ])


# ══════════════════════════════════════════════════════════════
# 1. COMPUTE
# ══════════════════════════════════════════════════════════════

def test_compute_positioning_basic(two_expiry_chain):
    """Hand-checked values for a 2-expiry chain."""
    r = compute_positioning(two_expiry_chain, date(2026, 5, 22))
    assert r.calls_total == 165.0    # (100+10) + (50+5)
    assert r.puts_total == 132.0     # (80+8) + (40+4)
    assert r.net == 33.0
    assert isinstance(r, PositioningResult)


def test_compute_positioning_calls_only():
    chain = pd.DataFrame([
        {"strike": 100.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 500, "volume": 50},
        {"strike": 105.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 300, "volume": 30},
    ])
    r = compute_positioning(chain, date(2026, 5, 22))
    assert r.calls_total == 880.0
    assert r.puts_total == 0.0
    assert r.net == 880.0


def test_compute_positioning_puts_only():
    chain = pd.DataFrame([
        {"strike": 100.0, "type": "P", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 700, "volume": 70},
    ])
    r = compute_positioning(chain, date(2026, 5, 22))
    assert r.calls_total == 0.0
    assert r.puts_total == 770.0
    assert r.net == -770.0


def test_compute_positioning_zero_strikes_excluded():
    """Strikes with zero OI and zero volume contribute nothing and don't error."""
    chain = pd.DataFrame([
        {"strike": 100.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 0, "volume": 0},  # excluded
        {"strike": 105.0, "type": "C", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 100, "volume": 10},
        {"strike": 110.0, "type": "P", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 0, "volume": 5},  # included (vol > 0)
        {"strike": 115.0, "type": "P", "expiry": pd.Timestamp("2026-05-22"),
         "oi": 50, "volume": 0},  # included (oi > 0)
    ])
    r = compute_positioning(chain, date(2026, 5, 22))
    assert r.calls_total == 110.0
    assert r.puts_total == 55.0
    assert r.net == 55.0


def test_compute_positioning_no_matching_expiry(two_expiry_chain, caplog):
    """No contracts for the requested expiry → zero result + warning logged."""
    with caplog.at_level(logging.WARNING, logger="positioning"):
        r = compute_positioning(two_expiry_chain, date(2026, 7, 4))
    assert r == PositioningResult(0.0, 0.0, 0.0)
    assert any("no contracts" in rec.message.lower() for rec in caplog.records)


def test_compute_positioning_all_expiries(two_expiry_chain):
    """Multi-expiry helper returns one PositioningResult per unique expiry."""
    out = compute_positioning_all_expiries(two_expiry_chain)
    assert set(out.keys()) == {date(2026, 5, 22), date(2026, 5, 29)}
    assert out[date(2026, 5, 22)].calls_total == 165.0
    assert out[date(2026, 5, 29)].calls_total == 330.0
    assert out[date(2026, 5, 29)].puts_total == 176.0
    for v in out.values():
        assert isinstance(v, PositioningResult)


# ══════════════════════════════════════════════════════════════
# 2. ARCHIVE
# ══════════════════════════════════════════════════════════════

def test_load_positioning_empty_archive(tmp_archive):
    """No file → empty DataFrame with the documented columns."""
    df = load_positioning("GC", date(2026, 5, 22), tmp_archive)
    assert df.empty
    assert list(df.columns) == ["snapshot_time", "expiry", "calls_total", "puts_total", "net"]


def test_load_positioning_dedupe(tmp_archive):
    """Duplicate (snapshot_time, expiry) rows collapse to the latest write."""
    snap = date(2026, 5, 22)

    # First write: t=10:00, one expiry
    archive_positioning({
        "snapshot_date": snap,
        "snapshot_time": time(10, 0),
        "product": "GC",
        "expiries": {date(2026, 5, 22): (100.0, 80.0, 20.0)},
    }, config=tmp_archive)

    # Second write at the SAME snapshot_time for the SAME expiry → overrides
    archive_positioning({
        "snapshot_date": snap,
        "snapshot_time": time(10, 0),
        "product": "GC",
        "expiries": {date(2026, 5, 22): (200.0, 150.0, 50.0)},
    }, config=tmp_archive)

    # Third write: different snapshot_time → adds a new row
    archive_positioning({
        "snapshot_date": snap,
        "snapshot_time": time(10, 30),
        "product": "GC",
        "expiries": {date(2026, 5, 22): (210.0, 155.0, 55.0)},
    }, config=tmp_archive)

    df = load_positioning("GC", snap, tmp_archive)
    assert len(df) == 2
    # First row is 10:00 with the SECOND write's values
    assert df.iloc[0]["snapshot_time"] == "10:00:00"
    assert df.iloc[0]["calls_total"] == 200.0
    assert df.iloc[0]["puts_total"] == 150.0
    assert df.iloc[1]["snapshot_time"] == "10:30:00"
    assert df.iloc[1]["calls_total"] == 210.0


def test_archive_positioning_round_trip(tmp_archive):
    """Write → read returns the documented schema with the slash-form product."""
    snap = date(2026, 5, 22)
    archive_positioning({
        "snapshot_date": snap,
        "snapshot_time": time(10, 30),
        "product": "/GC",  # leading slash is tolerated
        "expiries": {
            date(2026, 5, 22): (1000.0, 500.0, 500.0),
            date(2026, 5, 29): (700.0, 900.0, -200.0),
        },
    }, config=tmp_archive)

    path = get_positioning_path("GC", snap, tmp_archive)
    assert path.exists()
    # Slug-form path (no slash) — see brief Enhancement 3.
    assert "positioning" in path.parts and "GC" in path.parts

    df = load_positioning("GC", snap, tmp_archive)
    assert len(df) == 2
    assert set(df["expiry"]) == {date(2026, 5, 22), date(2026, 5, 29)}
    assert list_positioning_dates("GC", tmp_archive) == [snap]


# ══════════════════════════════════════════════════════════════
# 3. PIPELINE GATE
# ══════════════════════════════════════════════════════════════

def test_pipeline_skips_positioning_for_equities(tmp_archive, two_expiry_chain):
    """SPY (equity) must NOT produce a positioning archive entry."""
    out = maybe_archive_positioning(
        two_expiry_chain, PRODUCTS["SPY"], date(2026, 5, 22),
        snapshot_time_value=time(10, 30), config=tmp_archive,
    )
    assert out is None
    spy_path = Path(tmp_archive.root_dir) / "positioning" / "SPY"
    assert not spy_path.exists()

    # Sanity check: the same chain for GC DOES write.
    chain_for_gc = two_expiry_chain.copy()
    chain_for_gc["is_call"] = chain_for_gc["type"].eq("C")
    chain_for_gc = chain_for_gc.drop(columns=["type"])
    written = maybe_archive_positioning(
        chain_for_gc, PRODUCTS["GC"], date(2026, 5, 22),
        snapshot_time_value=time(10, 30), config=tmp_archive,
    )
    assert written is not None
    df = load_positioning("GC", date(2026, 5, 22), tmp_archive)
    assert not df.empty


# ══════════════════════════════════════════════════════════════
# 4. DASHBOARD HELPERS
# ══════════════════════════════════════════════════════════════

def test_dashboard_sums_multiple_expiries():
    """sum_positioning_across_expiries: 0, 1, 2, all expiries selected."""
    df = pd.DataFrame([
        {"snapshot_time": "10:00:00", "expiry": date(2026, 5, 18),
         "calls_total": 100, "puts_total": 80, "net": 20},
        {"snapshot_time": "10:00:00", "expiry": date(2026, 5, 19),
         "calls_total": 50, "puts_total": 30, "net": 20},
        {"snapshot_time": "10:30:00", "expiry": date(2026, 5, 18),
         "calls_total": 110, "puts_total": 85, "net": 25},
        {"snapshot_time": "10:30:00", "expiry": date(2026, 5, 19),
         "calls_total": 60, "puts_total": 35, "net": 25},
    ])

    # Nothing selected
    empty = sum_positioning_across_expiries(df, [])
    assert empty.empty

    # Single expiry
    one = sum_positioning_across_expiries(df, [date(2026, 5, 18)])
    assert len(one) == 2
    assert list(one["calls"]) == [100, 110]
    assert list(one["puts"]) == [80, 85]

    # Both expiries
    both = sum_positioning_across_expiries(df, [date(2026, 5, 18), date(2026, 5, 19)])
    assert list(both["calls"]) == [150, 170]
    assert list(both["puts"]) == [110, 120]


def test_current_week_default():
    """current_week_dates returns the Mon-Fri block of the latest date."""
    # Mon 2026-05-18 .. Fri 2026-05-22; latest is Wed 2026-05-20
    dates = [date(2026, 5, 11), date(2026, 5, 12), date(2026, 5, 18),
             date(2026, 5, 19), date(2026, 5, 20)]
    week = current_week_dates(dates)
    assert week == {
        date(2026, 5, 18), date(2026, 5, 19), date(2026, 5, 20),
        date(2026, 5, 21), date(2026, 5, 22),
    }
    # Empty input → empty set
    assert current_week_dates([]) == set()


def test_expiry_list_dynamic_to_trading_date(tmp_archive):
    """Different trading dates surface different expiries."""
    # Week 1: trading date 2026-05-18, expiries this week
    archive_positioning({
        "snapshot_date": date(2026, 5, 18),
        "snapshot_time": time(10, 30),
        "product": "GC",
        "expiries": {
            date(2026, 5, 18): (100.0, 80.0, 20.0),
            date(2026, 5, 22): (200.0, 150.0, 50.0),
        },
    }, config=tmp_archive)

    # Week 2: trading date 2026-05-25, expiries next week
    archive_positioning({
        "snapshot_date": date(2026, 5, 25),
        "snapshot_time": time(10, 30),
        "product": "GC",
        "expiries": {
            date(2026, 5, 25): (110.0, 85.0, 25.0),
            date(2026, 5, 29): (210.0, 160.0, 50.0),
        },
    }, config=tmp_archive)

    df1 = load_positioning("GC", date(2026, 5, 18), tmp_archive)
    df2 = load_positioning("GC", date(2026, 5, 25), tmp_archive)
    expiries_1 = sorted({pd.Timestamp(d).date() for d in df1["expiry"]})
    expiries_2 = sorted({pd.Timestamp(d).date() for d in df2["expiry"]})
    assert expiries_1 == [date(2026, 5, 18), date(2026, 5, 22)]
    assert expiries_2 == [date(2026, 5, 25), date(2026, 5, 29)]
    assert expiries_1 != expiries_2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
