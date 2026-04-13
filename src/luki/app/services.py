"""Shared runtime services for the LUKI Gradio app.

These are **module-level singletons** — instantiated once when the process
starts, and reused across every Gradio callback. Do NOT put this state in
`gr.State`: that would create a per-session copy of the DINOv3 model (1.2GB).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from luki.embeddings.dataset import load_manifest
from luki.embeddings.model import DinoV3Embedder
from luki.embeddings.store import QdrantStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_config(path: str = "config/base.yaml") -> dict:
    """Load the YAML config once."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_manifest() -> pd.DataFrame:
    """Load the ETL manifest once (parquet read is cheap, but still)."""
    cfg = get_config()
    manifest_path = Path(cfg["data"]["processed_dir"]).resolve() / "manifest.parquet"
    return load_manifest(manifest_path)


@lru_cache(maxsize=1)
def get_embedder() -> DinoV3Embedder:
    """Load DINOv3 once. This is the expensive singleton (~1.2GB weights)."""
    cfg = get_config()
    logger.info("Initializing DinoV3Embedder singleton...")
    return DinoV3Embedder(
        model_name=cfg["embeddings"]["model_name"],
        device=cfg["embeddings"].get("device", "auto"),
    )


@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    """Connect to Qdrant once. gRPC client is cheap to create."""
    cfg = get_config()
    logger.info("Connecting to QdrantStore singleton...")
    return QdrantStore(
        url=cfg["qdrant"]["url"],
        collection_name=cfg["qdrant"]["collection_name"],
        vector_size=cfg["qdrant"]["vector_size"],
    )


def warmup() -> None:
    """Force-instantiate every singleton on startup.

    Without this, the first user click would pay the 10-second model-load
    cost instead of an immediate answer. Eager warmup is the right pattern
    for user-facing ML services.
    """
    get_config()
    get_manifest()
    get_embedder()
    get_store()
    logger.info("All singletons initialized and ready.")
