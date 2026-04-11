"""Helpers to iterate over the ETL manifest in batches.

The manifest is a small parquet (205 rows today, maybe 100k tomorrow). We
don't need `torch.utils.data.DataLoader` machinery — a simple generator is
clearer, has no collate_fn edge cases, and keeps the pipeline synchronous.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# Columns we preserve in the Qdrant payload, minus `model_version` which is
# added by the pipeline. Keep this list small — payload bloat slows search.
PAYLOAD_COLUMNS: tuple[str, ...] = (
    "file_hash",
    "absolute_path",
    "filename",
    "medium",
    "camera",
    "year",
    "session_name",
    "roll_date",
    "film_stock",
    "film_iso",
    "datetime_original",
    "width",
    "height",
    "gps_lat",
    "gps_lon",
)


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Read the parquet manifest produced by the ETL pipeline."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run the ETL pipeline first "
            f"(python -m luki.etl.pipeline)."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded manifest: %d rows, %d columns", len(df), len(df.columns))
    return df


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Extract JSON-serializable payload from a manifest row."""
    payload: dict[str, Any] = {}
    for col in PAYLOAD_COLUMNS:
        if col not in row:
            continue
        val = row[col]
        # pandas represents missing values as NaN/NaT/None; Qdrant accepts
        # None but not NaN, so normalize.
        if pd.isna(val):
            payload[col] = None
        elif hasattr(val, "item"):
            # numpy scalar -> python scalar
            payload[col] = val.item()
        else:
            payload[col] = val
    return payload


def load_image(path: str | Path) -> Image.Image:
    """Open an image and convert to RGB. Raises on failure."""
    img = Image.open(path)
    # `load()` forces decoding now, so corrupted files fail here instead of
    # deep inside the embedder.
    img.load()
    return img.convert("RGB")


def iter_batches(
    df: pd.DataFrame,
    batch_size: int,
) -> Iterator[tuple[list[dict[str, Any]], list[Image.Image], list[dict[str, Any]]]]:
    """Yield (records, loaded_images, payloads) tuples of size <= batch_size.

    Corrupted images are skipped in place and NOT included in the yielded
    batch. The caller is responsible for counting `failed` via the logs.
    """
    records: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    payloads: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        try:
            img = load_image(row["absolute_path"])
        except Exception as exc:
            logger.warning(
                "Skipping corrupted/unreadable image %s: %s",
                row.get("absolute_path", "<unknown>"),
                exc,
            )
            continue

        records.append({"file_hash": row["file_hash"]})
        images.append(img)
        payloads.append(_row_to_payload(row))

        if len(images) >= batch_size:
            yield records, images, payloads
            records, images, payloads = [], [], []

    if images:
        yield records, images, payloads
