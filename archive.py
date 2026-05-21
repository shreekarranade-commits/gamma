"""
Data lifecycle management: archive, purge, replay.

Two-tier storage:
  Tier 1: Raw chain (Parquet) - full replay capability
  Tier 2: Computed results (Parquet + JSON) - fast retrieval

Positioning archive (v1.4): separate per-day intraday Parquet under
  archive/positioning/{product}/{date}/positioning.parquet
Independent of the two-tier Greeks archive — kept indefinitely.
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from config import ArchiveConfig, ARCHIVE_DEFAULTS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# ARCHIVE WRITE
# ══════════════════════════════════════════════════════════════

def get_archive_path(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> Path:
    """Get the archive directory for a product/date combination."""
    return Path(config.root_dir) / product / snapshot_date.isoformat()


def archive_results(pipeline_result: dict, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> Path:
    """
    Write pipeline results to the archive.

    Creates:
      archive/{product}/{date}/
        raw_chain.parquet          (Tier 1)
        computed_greeks.parquet    (Tier 2)
        strike_profiles.parquet   (Tier 2)
        expiry_breakdown.parquet  (Tier 2)
        scores.json               (Tier 2)
        metadata.json             (Tier 2)
    """
    product = pipeline_result["metadata"]["product"]
    snapshot_date = pipeline_result["snapshot_date"]
    archive_dir = get_archive_path(product, snapshot_date, config)
    archive_dir.mkdir(parents=True, exist_ok=True)

    contracts_df = pipeline_result["contracts"]
    profiles_df = pipeline_result["strike_profiles"]
    breakdown_df = pipeline_result["expiry_breakdown"]
    scores = pipeline_result["scores"]
    metadata = pipeline_result["metadata"]

    # Tier 1: Raw chain (contracts before aggregation)
    if len(contracts_df) > 0:
        # Prepare for parquet: convert datetime columns
        raw = contracts_df.copy()
        if "expiry" in raw.columns:
            raw["expiry"] = raw["expiry"].astype(str)
        raw.to_parquet(archive_dir / "raw_chain.parquet", index=False)
        logger.info(f"  Archived Tier 1: raw_chain.parquet ({len(raw)} contracts)")

    # Tier 2: Computed Greeks per contract
    if len(contracts_df) > 0:
        greeks = contracts_df[
            [c for c in contracts_df.columns
             if c in ["strike", "expiry", "is_call", "dte", "oi", "volume",
                       "volume_to_oi_ratio", "iv",
                       "delta", "gamma", "vanna", "charm", "gex", "vex", "cex"]]
        ].copy()
        if "expiry" in greeks.columns:
            greeks["expiry"] = greeks["expiry"].astype(str)
        greeks.to_parquet(archive_dir / "computed_greeks.parquet", index=False)

    # Tier 2: Strike profiles
    if len(profiles_df) > 0:
        profiles_df.to_parquet(archive_dir / "strike_profiles.parquet", index=False)

    # Tier 2: Expiry breakdown
    if len(breakdown_df) > 0:
        breakdown_df.to_parquet(archive_dir / "expiry_breakdown.parquet", index=False)

    # Tier 2: Scores
    with open(archive_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    # Metadata
    with open(archive_dir / "metadata.json", "w") as f:
        # Convert non-serializable types
        meta_safe = {}
        for k, v in metadata.items():
            if isinstance(v, (date, datetime)):
                meta_safe[k] = v.isoformat()
            else:
                meta_safe[k] = v
        json.dump(meta_safe, f, indent=2)

    # Update manifest
    _update_manifest(config)

    logger.info(f"  Archive written: {archive_dir}")
    return archive_dir


# ══════════════════════════════════════════════════════════════
# ARCHIVE READ
# ══════════════════════════════════════════════════════════════

def load_scores(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> dict:
    """Load headline scores from archive."""
    path = get_archive_path(product, snapshot_date, config) / "scores.json"
    if not path.exists():
        raise FileNotFoundError(f"No archived scores for {product} on {snapshot_date}")
    with open(path) as f:
        return json.load(f)


def load_metadata(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> dict:
    """Load metadata from archive."""
    path = get_archive_path(product, snapshot_date, config) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No archived metadata for {product} on {snapshot_date}")
    with open(path) as f:
        return json.load(f)


def load_strike_profiles(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> pd.DataFrame:
    """Load strike profiles from archive."""
    path = get_archive_path(product, snapshot_date, config) / "strike_profiles.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No archived profiles for {product} on {snapshot_date}")
    return pd.read_parquet(path)


def load_expiry_breakdown(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> pd.DataFrame:
    """Load expiry breakdown from archive."""
    path = get_archive_path(product, snapshot_date, config) / "expiry_breakdown.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No archived breakdown for {product} on {snapshot_date}")
    return pd.read_parquet(path)


def load_contracts(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> pd.DataFrame:
    """Load per-contract data from archive."""
    path = get_archive_path(product, snapshot_date, config) / "computed_greeks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No archived contracts for {product} on {snapshot_date}")
    return pd.read_parquet(path)


def load_raw_chain(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> pd.DataFrame:
    """Load raw chain (Tier 1) from archive."""
    path = get_archive_path(product, snapshot_date, config) / "raw_chain.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No archived raw chain for {product} on {snapshot_date}")
    return pd.read_parquet(path)


# ══════════════════════════════════════════════════════════════
# AVAILABLE DATES
# ══════════════════════════════════════════════════════════════

def load_score_history(product: str, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> pd.DataFrame:
    """
    Iterate every archived date for a product and load scores.json plus the
    underlying price from metadata.json.

    Returns a DataFrame with columns:
        date, gex, vex, cex, gex_plus, underlying_price,
        gex_flip, vex_flip, cex_flip
    Sorted ascending by date. Empty DataFrame if no archives exist.
    """
    rows = []
    for d in list_archived_dates(product, config):
        try:
            scores = load_scores(product, d, config)
        except FileNotFoundError:
            continue
        try:
            meta = load_metadata(product, d, config)
            underlying = meta.get("underlying_price")
        except FileNotFoundError:
            underlying = None

        rows.append({
            "date": d,
            "gex": float(scores.get("gex", 0)),
            "vex": float(scores.get("vex", 0)),
            "cex": float(scores.get("cex", 0)),
            "gex_plus": float(scores.get("gex_plus", 0)),
            "underlying_price": underlying,
            "gex_flip": scores.get("gex_flip") or [],
            "vex_flip": scores.get("vex_flip") or [],
            "cex_flip": scores.get("cex_flip") or [],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "gex", "vex", "cex", "gex_plus", "underlying_price",
            "gex_flip", "vex_flip", "cex_flip",
        ])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def list_archived_dates(product: str, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> list[date]:
    """List all archived dates for a product, sorted ascending."""
    product_dir = Path(config.root_dir) / product
    if not product_dir.exists():
        return []
    dates = []
    for d in product_dir.iterdir():
        if d.is_dir():
            try:
                dates.append(date.fromisoformat(d.name))
            except ValueError:
                continue
    return sorted(dates)


def list_archived_products(config: ArchiveConfig = ARCHIVE_DEFAULTS) -> list[str]:
    """List all products with archived data."""
    root = Path(config.root_dir)
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir() and d.name != "replay"])


def get_archive_availability(product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS) -> dict:
    """Check what tiers are available for a product/date."""
    archive_dir = get_archive_path(product, snapshot_date, config)
    return {
        "tier1": (archive_dir / "raw_chain.parquet").exists(),
        "tier2_greeks": (archive_dir / "computed_greeks.parquet").exists(),
        "tier2_profiles": (archive_dir / "strike_profiles.parquet").exists(),
        "tier2_scores": (archive_dir / "scores.json").exists(),
        "tier2_metadata": (archive_dir / "metadata.json").exists(),
    }


# ══════════════════════════════════════════════════════════════
# PURGE
# ══════════════════════════════════════════════════════════════

def purge(
    product: str = None,
    before_date: date = None,
    tier: str = "all",
    dry_run: bool = True,
    config: ArchiveConfig = ARCHIVE_DEFAULTS,
) -> dict:
    """
    Purge archived data.

    Parameters
    ----------
    product : str or None - specific product or all
    before_date : date - purge data older than this date
    tier : str - "1", "2", or "all"
    dry_run : bool - if True, report what would be deleted without deleting
    config : ArchiveConfig

    Returns
    -------
    dict with purge summary
    """
    root = Path(config.root_dir)
    products = [product] if product else list_archived_products(config)

    files_to_delete = []
    dirs_to_delete = []

    tier1_files = ["raw_chain.parquet"]
    tier2_files = ["computed_greeks.parquet", "strike_profiles.parquet",
                   "expiry_breakdown.parquet", "scores.json", "metadata.json"]

    for prod in products:
        for archived_date in list_archived_dates(prod, config):
            if before_date and archived_date >= before_date:
                continue

            archive_dir = get_archive_path(prod, archived_date, config)

            if tier in ("1", "all"):
                for f in tier1_files:
                    fp = archive_dir / f
                    if fp.exists():
                        files_to_delete.append(fp)

            if tier in ("2", "all"):
                for f in tier2_files:
                    fp = archive_dir / f
                    if fp.exists():
                        files_to_delete.append(fp)

            # Check if directory will be empty after purge
            if tier == "all":
                dirs_to_delete.append(archive_dir)

    total_bytes = sum(f.stat().st_size for f in files_to_delete if f.exists())

    summary = {
        "files_count": len(files_to_delete),
        "dirs_count": len(dirs_to_delete),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info(f"  [DRY RUN] Would delete {len(files_to_delete)} files ({summary['total_mb']} MB)")
        for f in files_to_delete[:10]:
            logger.info(f"    {f}")
        if len(files_to_delete) > 10:
            logger.info(f"    ... and {len(files_to_delete) - 10} more")
    else:
        for f in files_to_delete:
            if f.exists():
                f.unlink()
        for d in dirs_to_delete:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        logger.info(f"  Purged {len(files_to_delete)} files ({summary['total_mb']} MB)")
        _update_manifest(config)

    return summary


def auto_purge(config: ArchiveConfig = ARCHIVE_DEFAULTS) -> dict:
    """Run automatic purge based on retention policies."""
    results = {}

    # Tier 1 purge
    if config.retention_tier1_days > 0:
        cutoff = date.today() - timedelta(days=config.retention_tier1_days)
        results["tier1"] = purge(
            before_date=cutoff, tier="1", dry_run=False, config=config
        )

    # Tier 2 purge
    if config.retention_tier2_days > 0:
        cutoff = date.today() - timedelta(days=config.retention_tier2_days)
        results["tier2"] = purge(
            before_date=cutoff, tier="2", dry_run=False, config=config
        )

    return results


# ══════════════════════════════════════════════════════════════
# MANIFEST
# ══════════════════════════════════════════════════════════════

def _update_manifest(config: ArchiveConfig = ARCHIVE_DEFAULTS):
    """Update the archive manifest file."""
    root = Path(config.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    products = list_archived_products(config)
    manifest = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "products": {},
        "total_size_bytes": 0,
    }

    for prod in products:
        dates = list_archived_dates(prod, config)
        prod_size = 0
        for d in dates:
            archive_dir = get_archive_path(prod, d, config)
            for f in archive_dir.iterdir():
                if f.is_file():
                    prod_size += f.stat().st_size

        manifest["products"][prod] = {
            "date_count": len(dates),
            "earliest": dates[0].isoformat() if dates else None,
            "latest": dates[-1].isoformat() if dates else None,
            "size_bytes": prod_size,
            "size_mb": round(prod_size / (1024 * 1024), 2),
        }
        manifest["total_size_bytes"] += prod_size

    manifest["total_size_mb"] = round(manifest["total_size_bytes"] / (1024 * 1024), 2)

    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ══════════════════════════════════════════════════════════════
# POSITIONING ARCHIVE (v1.4)
# ══════════════════════════════════════════════════════════════

POSITIONING_SUBDIR = "positioning"
POSITIONING_COLUMNS = [
    "snapshot_date", "snapshot_time", "product",
    "expiry", "calls_total", "puts_total", "net",
]


def _product_slug(product: str) -> str:
    """Strip the optional display slash: '/GC' → 'GC'."""
    return product.lstrip("/")


def get_positioning_path(
    product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS
) -> Path:
    """Path to a single trading day's positioning parquet."""
    return (
        Path(config.root_dir) / POSITIONING_SUBDIR
        / _product_slug(product) / snapshot_date.isoformat()
        / "positioning.parquet"
    )


def archive_positioning(
    record: dict, config: ArchiveConfig = ARCHIVE_DEFAULTS
) -> Path:
    """
    Append per-expiry positioning rows to the trading day's parquet.

    record keys:
      snapshot_date : date
      snapshot_time : time   — wall-clock ET, seconds zeroed
      product       : str    — "GC", "/GC", "CL", "/CL"
      expiries      : dict[date, tuple(calls_total, puts_total, net)]

    Last-write-wins on (snapshot_time, expiry) duplicates — manual CLI
    + scheduler overlaps can produce duplicate ticks; the latest write
    is kept and earlier rows for the same key dropped on read and write.
    """
    product_in = record["product"]
    slug = _product_slug(product_in)
    display = product_in if product_in.startswith("/") else f"/{product_in}"
    snapshot_date = record["snapshot_date"]
    snapshot_time = record["snapshot_time"]
    expiries = record.get("expiries") or {}

    path = get_positioning_path(slug, snapshot_date, config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(snapshot_time, time):
        snapshot_time_str = snapshot_time.isoformat(timespec="seconds")
    else:
        snapshot_time_str = str(snapshot_time)

    rows = []
    for expiry, totals in expiries.items():
        calls_total, puts_total, net = float(totals[0]), float(totals[1]), float(totals[2])
        rows.append({
            "snapshot_date": snapshot_date,
            "snapshot_time": snapshot_time_str,
            "product": display,
            "expiry": expiry,
            "calls_total": calls_total,
            "puts_total": puts_total,
            "net": net,
        })

    if not rows:
        logger.info(f"  Positioning: no expiries to archive for {display} on {snapshot_date}")
        return path

    new_df = pd.DataFrame(rows, columns=POSITIONING_COLUMNS)

    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"  Positioning: existing archive unreadable ({e}); overwriting")
            combined = new_df
    else:
        combined = new_df

    combined["expiry"] = pd.to_datetime(combined["expiry"]).dt.date
    combined = combined.drop_duplicates(
        subset=["snapshot_time", "expiry"], keep="last"
    ).reset_index(drop=True)
    combined = combined.sort_values(["snapshot_time", "expiry"]).reset_index(drop=True)

    combined.to_parquet(path, index=False)
    logger.info(
        f"  Positioning archived: {path} "
        f"({len(new_df)} new rows, {len(combined)} total in file)"
    )
    return path


def load_positioning(
    product: str, snapshot_date: date, config: ArchiveConfig = ARCHIVE_DEFAULTS
) -> pd.DataFrame:
    """
    Load all intraday positioning snapshots for a single trading day.

    Returns DataFrame with columns:
        snapshot_time, expiry, calls_total, puts_total, net.
    Empty DataFrame with those columns if no archive exists.

    Duplicates on (snapshot_time, expiry) are deduplicated on read,
    keeping the latest entry (defensive — writes already dedupe).
    """
    cols = ["snapshot_time", "expiry", "calls_total", "puts_total", "net"]
    path = get_positioning_path(product, snapshot_date, config)
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df = df.drop_duplicates(
        subset=["snapshot_time", "expiry"], keep="last"
    ).reset_index(drop=True)
    df = df.sort_values(["snapshot_time", "expiry"]).reset_index(drop=True)
    return df[cols].copy()


def list_positioning_dates(
    product: str, config: ArchiveConfig = ARCHIVE_DEFAULTS
) -> list[date]:
    """List archived positioning dates for a product, ascending."""
    base = Path(config.root_dir) / POSITIONING_SUBDIR / _product_slug(product)
    if not base.exists():
        return []
    out = []
    for sub in base.iterdir():
        if sub.is_dir() and (sub / "positioning.parquet").exists():
            try:
                out.append(date.fromisoformat(sub.name))
            except ValueError:
                continue
    return sorted(out)
