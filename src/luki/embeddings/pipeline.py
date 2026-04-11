"""Orchestrator for the LUKI embeddings pipeline.

End-to-end flow:
    manifest.parquet --> filter (cache) --> batched forward pass --> Qdrant

The pipeline is resumable and idempotent: re-running it only processes photos
that either (a) are missing from Qdrant or (b) were embedded with a different
`model_version`. `force=True` bypasses the cache entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from luki.embeddings.dataset import iter_batches, load_manifest
from luki.embeddings.model import DinoV3Embedder
from luki.embeddings.store import QdrantStore

logger = logging.getLogger(__name__)


def _manifest_path(config: dict[str, Any]) -> Path:
    return Path(config["data"]["processed_dir"]).resolve() / "manifest.parquet"


def _filter_already_embedded(
    df: pd.DataFrame,
    store: QdrantStore,
    model_version: str,
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose file_hash is already in Qdrant with the current model version."""
    all_hashes = df["file_hash"].tolist()
    existing = store.existing_hashes_for_version(all_hashes, model_version)
    if not existing:
        return df, 0
    mask = ~df["file_hash"].isin(existing)
    return df[mask].reset_index(drop=True), len(existing)


def run_embeddings(
    config: dict[str, Any],
    *,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Run the embeddings pipeline end-to-end.

    Returns a stats dict: {processed, skipped, failed, total}.
    """
    emb_cfg = config["embeddings"]
    qdr_cfg = config["qdrant"]

    # 1. Load manifest
    manifest_path = _manifest_path(config)
    df = load_manifest(manifest_path)
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
        logger.info("Limited manifest to first %d rows", limit)
    total = len(df)

    # 2. Instantiate the embedder and vector store
    embedder = DinoV3Embedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "auto"),
    )
    store = QdrantStore(
        url=qdr_cfg["url"],
        collection_name=qdr_cfg["collection_name"],
        vector_size=qdr_cfg["vector_size"],
        distance=qdr_cfg.get("distance", "Cosine"),
    )

    # Sanity check: embedder dim must match store dim
    if embedder.hidden_size != store.vector_size:
        raise ValueError(
            f"Embedder produces {embedder.hidden_size}-dim vectors but the "
            f"store collection expects {store.vector_size}. Update config."
        )

    # 3. Cache filter
    if force:
        logger.info("--force: bypassing cache, re-embedding all photos")
        skipped = 0
        to_process = df
    else:
        to_process, skipped = _filter_already_embedded(
            df, store, embedder.model_version
        )
        if skipped:
            logger.info(
                "Cache hit: %d/%d photos already embedded with current model",
                skipped,
                total,
            )

    if to_process.empty:
        logger.info("Nothing to do. All %d photos are already up to date.", total)
        return {"processed": 0, "skipped": skipped, "failed": 0, "total": total}

    # 4. Process in batches
    normalize = emb_cfg.get("normalize", True)
    batch_size = int(emb_cfg.get("batch_size", 8))
    processed = 0
    failed = total - len(to_process) - skipped  # corrupted rows (caught in iter_batches)
    # Re-count failed by watching what iter_batches actually yields
    total_to_iter = len(to_process)
    yielded = 0

    with tqdm(total=total_to_iter, desc="Embedding", unit="img") as pbar:
        for records, images, payloads in iter_batches(to_process, batch_size=batch_size):
            yielded += len(images)
            file_hashes = [r["file_hash"] for r in records]

            vectors = embedder.embed(images, normalize=normalize)

            # Stamp model_version into every payload for cache invalidation
            for p in payloads:
                p["model_version"] = embedder.model_version

            store.upsert_batch(file_hashes, vectors, payloads)
            processed += len(images)
            pbar.update(len(images))

    failed = total_to_iter - yielded

    logger.info(
        "Done. processed=%d skipped=%d failed=%d total=%d",
        processed,
        skipped,
        failed,
        total,
    )
    return {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total": total,
    }
