"""
Tests for archive.py: write, read, list, purge, manifest.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from archive import (
    archive_results, load_scores, load_metadata, load_strike_profiles,
    load_expiry_breakdown, load_contracts, load_raw_chain,
    list_archived_products, list_archived_dates, get_archive_availability,
    purge, get_archive_path, load_score_history,
)
from config import ArchiveConfig
from pipeline import run_pipeline_synthetic


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_archive(tmp_path):
    """Temporary archive directory."""
    return ArchiveConfig(root_dir=str(tmp_path / "archive"))


@pytest.fixture
def archived_result(tmp_archive):
    """Run synthetic pipeline and archive the result."""
    result = run_pipeline_synthetic("SPY", 550.0, 0.05)
    archive_results(result, config=tmp_archive)
    return result, tmp_archive


# ══════════════════════════════════════════════════════════════
# 1. ARCHIVE WRITE
# ══════════════════════════════════════════════════════════════

class TestArchiveWrite:

    def test_creates_directory_structure(self, archived_result):
        result, config = archived_result
        archive_dir = get_archive_path("SPY", result["snapshot_date"], config)
        assert archive_dir.exists()

    def test_all_files_created(self, archived_result):
        result, config = archived_result
        archive_dir = get_archive_path("SPY", result["snapshot_date"], config)
        expected_files = [
            "raw_chain.parquet",
            "computed_greeks.parquet",
            "strike_profiles.parquet",
            "expiry_breakdown.parquet",
            "scores.json",
            "metadata.json",
        ]
        for f in expected_files:
            assert (archive_dir / f).exists(), f"Missing file: {f}"

    def test_manifest_created(self, archived_result):
        _, config = archived_result
        manifest_path = Path(config.root_dir) / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_has_product(self, archived_result):
        _, config = archived_result
        with open(Path(config.root_dir) / "manifest.json") as f:
            manifest = json.load(f)
        assert "SPY" in manifest["products"]
        assert manifest["products"]["SPY"]["date_count"] == 1

    def test_overwrite_on_rerun(self, archived_result):
        """Re-archiving same product/date overwrites."""
        result, config = archived_result
        archive_dir = get_archive_path("SPY", result["snapshot_date"], config)
        mtime_1 = (archive_dir / "scores.json").stat().st_mtime

        # Re-archive
        import time
        time.sleep(0.1)
        archive_results(result, config=config)
        mtime_2 = (archive_dir / "scores.json").stat().st_mtime
        assert mtime_2 > mtime_1


# ══════════════════════════════════════════════════════════════
# 2. ARCHIVE READ
# ══════════════════════════════════════════════════════════════

class TestArchiveRead:

    def test_load_scores(self, archived_result):
        result, config = archived_result
        scores = load_scores("SPY", result["snapshot_date"], config)
        assert scores["gex"] == result["scores"]["gex"]
        assert scores["vex"] == result["scores"]["vex"]
        assert scores["cex"] == result["scores"]["cex"]

    def test_load_metadata(self, archived_result):
        result, config = archived_result
        meta = load_metadata("SPY", result["snapshot_date"], config)
        assert meta["product"] == "SPY"
        assert meta["underlying_price"] == 550.0

    def test_load_strike_profiles(self, archived_result):
        result, config = archived_result
        profiles = load_strike_profiles("SPY", result["snapshot_date"], config)
        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) > 0
        assert "strike" in profiles.columns
        assert "gex" in profiles.columns

    def test_load_expiry_breakdown(self, archived_result):
        result, config = archived_result
        breakdown = load_expiry_breakdown("SPY", result["snapshot_date"], config)
        assert isinstance(breakdown, pd.DataFrame)
        assert "expiry_bucket" in breakdown.columns

    def test_load_contracts(self, archived_result):
        result, config = archived_result
        contracts = load_contracts("SPY", result["snapshot_date"], config)
        assert isinstance(contracts, pd.DataFrame)
        assert "gamma" in contracts.columns

    def test_load_raw_chain(self, archived_result):
        result, config = archived_result
        raw = load_raw_chain("SPY", result["snapshot_date"], config)
        assert isinstance(raw, pd.DataFrame)
        assert len(raw) > 0

    def test_missing_product_raises(self, tmp_archive):
        with pytest.raises(FileNotFoundError):
            load_scores("FAKE", date(2026, 1, 1), tmp_archive)

    def test_missing_date_raises(self, archived_result):
        _, config = archived_result
        with pytest.raises(FileNotFoundError):
            load_scores("SPY", date(2020, 1, 1), config)


# ══════════════════════════════════════════════════════════════
# 3. LISTING
# ══════════════════════════════════════════════════════════════

class TestArchiveListing:

    def test_list_products(self, archived_result):
        _, config = archived_result
        products = list_archived_products(config)
        assert "SPY" in products

    def test_list_dates(self, archived_result):
        result, config = archived_result
        dates = list_archived_dates("SPY", config)
        assert result["snapshot_date"] in dates

    def test_dates_sorted(self, tmp_archive):
        """Multiple dates are returned sorted ascending."""
        # Archive three different dates
        for price, day_offset in [(550, 0), (551, 1), (549, 2)]:
            r = run_pipeline_synthetic("SPY", float(price), 0.05)
            from datetime import timedelta
            r["snapshot_date"] = date.today() - timedelta(days=day_offset)
            archive_results(r, config=tmp_archive)

        dates = list_archived_dates("SPY", tmp_archive)
        assert dates == sorted(dates)

    def test_availability_check(self, archived_result):
        result, config = archived_result
        avail = get_archive_availability("SPY", result["snapshot_date"], config)
        assert avail["tier1"] is True
        assert avail["tier2_scores"] is True
        assert avail["tier2_profiles"] is True

    def test_availability_missing_date(self, tmp_archive):
        avail = get_archive_availability("SPY", date(2020, 1, 1), tmp_archive)
        assert avail["tier1"] is False
        assert avail["tier2_scores"] is False


# ══════════════════════════════════════════════════════════════
# 4. PURGE
# ══════════════════════════════════════════════════════════════

class TestPurge:

    def test_dry_run_no_delete(self, archived_result):
        result, config = archived_result
        summary = purge(
            product="SPY",
            before_date=date(2030, 1, 1),
            tier="all",
            dry_run=True,
            config=config,
        )
        assert summary["dry_run"] is True
        assert summary["files_count"] > 0
        # Files should still exist
        avail = get_archive_availability("SPY", result["snapshot_date"], config)
        assert avail["tier1"] is True

    def test_force_delete_tier1(self, tmp_archive):
        """Purge tier 1 only, keep tier 2."""
        r = run_pipeline_synthetic("SPY", 550.0, 0.05)
        archive_results(r, config=tmp_archive)

        purge(
            product="SPY",
            before_date=date(2030, 1, 1),
            tier="1",
            dry_run=False,
            config=tmp_archive,
        )

        avail = get_archive_availability("SPY", r["snapshot_date"], tmp_archive)
        assert avail["tier1"] is False      # deleted
        assert avail["tier2_scores"] is True  # preserved

    def test_force_delete_all(self, tmp_archive):
        """Purge all tiers."""
        r = run_pipeline_synthetic("SPY", 550.0, 0.05)
        archive_results(r, config=tmp_archive)

        purge(
            product="SPY",
            before_date=date(2030, 1, 1),
            tier="all",
            dry_run=False,
            config=tmp_archive,
        )

        avail = get_archive_availability("SPY", r["snapshot_date"], tmp_archive)
        assert avail["tier1"] is False
        assert avail["tier2_scores"] is False

    def test_purge_respects_date_cutoff(self, tmp_archive):
        """Only data before the cutoff date is purged."""
        r = run_pipeline_synthetic("SPY", 550.0, 0.05)
        archive_results(r, config=tmp_archive)

        # Purge before yesterday — today's data should survive
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        purge(
            product="SPY",
            before_date=yesterday,
            tier="all",
            dry_run=False,
            config=tmp_archive,
        )

        avail = get_archive_availability("SPY", r["snapshot_date"], tmp_archive)
        assert avail["tier1"] is True  # Not purged (today >= yesterday)


# ══════════════════════════════════════════════════════════════
# SCORE HISTORY (Enhancement 7, v1.3)
# ══════════════════════════════════════════════════════════════

class TestScoreHistory:

    def _stamp_archive(self, tmp_archive, product, snapshot_date,
                        underlying, gex, vex, cex):
        """Hand-craft a minimal archive entry to control date and values."""
        from archive import get_archive_path
        d = get_archive_path(product, snapshot_date, tmp_archive)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "scores.json", "w") as f:
            json.dump({
                "gex": gex, "vex": vex, "cex": cex, "gex_plus": gex + vex,
                "gex_flip": [550.0], "vex_flip": [], "cex_flip": [],
            }, f)
        with open(d / "metadata.json", "w") as f:
            json.dump({
                "product": product, "underlying_price": underlying,
                "engine_version": "1.3",
            }, f)

    def test_empty_when_no_archive(self, tmp_archive):
        df = load_score_history("SPY", config=tmp_archive)
        assert df.empty
        assert list(df.columns) == [
            "date", "gex", "vex", "cex", "gex_plus", "underlying_price",
            "gex_flip", "vex_flip", "cex_flip",
        ]

    def test_loads_multiple_dates_sorted(self, tmp_archive):
        from datetime import date as d
        # Stamp three out-of-order dates.
        self._stamp_archive(tmp_archive, "SPY", d(2026, 4, 22), 549.0, 100.0, -50.0, 5.0)
        self._stamp_archive(tmp_archive, "SPY", d(2026, 4, 24), 552.0, 200.0, -80.0, 7.0)
        self._stamp_archive(tmp_archive, "SPY", d(2026, 4, 23), 551.0, 150.0, -60.0, 6.0)

        df = load_score_history("SPY", config=tmp_archive)
        assert len(df) == 3
        # Sorted ascending by date.
        assert list(df["date"]) == [d(2026, 4, 22), d(2026, 4, 23), d(2026, 4, 24)]
        # Values pulled correctly.
        assert df.iloc[1]["gex"] == 150.0
        assert df.iloc[1]["underlying_price"] == 551.0
        # gex_plus = gex + vex.
        assert df.iloc[2]["gex_plus"] == 200.0 + (-80.0)
        # Flip strikes round-trip as a list.
        assert df.iloc[0]["gex_flip"] == [550.0]

    def test_synthetic_archive_compatible(self, archived_result):
        """A real archive_results() write is loadable via load_score_history."""
        result, config = archived_result
        df = load_score_history("SPY", config=config)
        assert len(df) >= 1
        assert "gex_flip" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
