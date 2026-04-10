"""
pipeline.py

Orchestrates the full ETL pipeline:
    discover → parse path → extract metadata → persist manifest

Output:
    data/processed/manifest.parquet   — full dataset, one row per photo
    data/processed/manifest_summary.json — run statistics
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from luki.etl.discover import discover_images
from luki.etl.extract import extract_metadata
from luki.etl.path_parser import parse_photo_path, PhotoPath

logger = logging.getLogger(__name__)


def run_etl(config: dict) -> pd.DataFrame:
    """
    Runs the full ETL pipeline from raw photos to a structured manifest.

    Args:
        config: loaded configuration dict (from base.yaml or colab.yaml)

    Returns:
        DataFrame with one row per successfully processed photo.
        Also persists manifest.parquet and manifest_summary.json to disk.
    """
    raw_dir = Path(config["data"]["raw_dir"]).resolve()
    processed_dir = Path(config["data"]["processed_dir"]).resolve()
    extensions = set(config["data"]["supported_extensions"])

    processed_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("LUKI ETL pipeline starting")
    logger.info(f"  Source : {raw_dir}")
    logger.info(f"  Output : {processed_dir}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1 — Discover
    # ------------------------------------------------------------------ #
    logger.info("Step 1/3 — Discovering images...")
    image_paths = discover_images(raw_dir, extensions)

    if not image_paths:
        logger.warning("No images found. Check your raw_dir path and folder structure.")
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Step 2 — Extract metadata
    # ------------------------------------------------------------------ #
    logger.info(f"Step 2/3 — Extracting metadata from {len(image_paths)} images...")

    records: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for path in tqdm(image_paths, desc="Extracting", unit="img"):

        # Parse folder structure
        try:
            photo_path: PhotoPath = parse_photo_path(path, raw_dir)
        except ValueError as exc:
            logger.warning(f"Path convention error — skipping {path}: {exc}")
            failed.append({"path": str(path), "reason": f"path_parse: {exc}"})
            continue

        # Extract file + EXIF metadata
        file_meta = extract_metadata(path)
        if file_meta is None:
            logger.warning(f"Could not read image — skipping {path}")
            failed.append({"path": str(path), "reason": "unreadable_image"})
            continue

        # Merge path metadata + file metadata into a single flat record
        record = {
            # From folder structure
            "medium": photo_path.medium,
            "year": photo_path.year,
            "camera": photo_path.camera,
            "session_name": photo_path.session_name,
            "roll_date": photo_path.roll_date,
            "film_stock": photo_path.film_stock,
            "film_iso": photo_path.film_iso,
            "roll_tags": photo_path.roll_tags if photo_path.roll_tags else None,
            # From file + EXIF
            **file_meta,
        }
        records.append(record)

    # ------------------------------------------------------------------ #
    # Step 3 — Persist
    # ------------------------------------------------------------------ #
    logger.info("Step 3/3 — Persisting manifest...")

    df = pd.DataFrame(records)

    if not df.empty:
        # Type cleanup
        df["year"] = df["year"].astype("Int64")
        df["film_iso"] = df["film_iso"].astype("Int64")
        df["size_bytes"] = df["size_bytes"].astype("Int64")
        df["width"] = df["width"].astype("Int64")
        df["height"] = df["height"].astype("Int64")

        # Convert roll_tags list → pipe-separated string for parquet compatibility
        df["roll_tags"] = df["roll_tags"].apply(
            lambda tags: "|".join(tags) if isinstance(tags, list) else None
        )

        manifest_path = processed_dir / "manifest.parquet"
        df.to_parquet(manifest_path, index=False)
        logger.info(f"Manifest saved → {manifest_path}")

    # Summary
    elapsed = time.time() - start_time
    summary = {
        "total_discovered": len(image_paths),
        "total_processed": len(records),
        "total_failed": len(failed),
        "success_rate_pct": round(len(records) / len(image_paths) * 100, 1) if image_paths else 0,
        "elapsed_seconds": round(elapsed, 2),
        "failed_files": failed,
    }

    summary_path = processed_dir / "manifest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    return df


def _print_summary(summary: dict) -> None:
    logger.info("=" * 60)
    logger.info("ETL complete")
    logger.info(f"  Discovered : {summary['total_discovered']}")
    logger.info(f"  Processed  : {summary['total_processed']}")
    logger.info(f"  Failed     : {summary['total_failed']}")
    logger.info(f"  Success    : {summary['success_rate_pct']}%")
    logger.info(f"  Time       : {summary['elapsed_seconds']}s")
    logger.info("=" * 60)

    if summary["failed_files"]:
        logger.warning("Failed files:")
        for entry in summary["failed_files"]:
            logger.warning(f"  {entry['path']} → {entry['reason']}")


if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    with open("config/base.yaml") as f:
        config = yaml.safe_load(f)

    df = run_etl(config)

    if not df.empty:
        print(f"\nManifest shape: {df.shape}")
        print(df.dtypes)
        print(df.head())
