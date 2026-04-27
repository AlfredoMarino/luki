"""Shared runtime services for the LUKI API layer.

These are **module-level singletons** — instantiated once when the process
starts, and reused across every request handler. Do NOT put this state in a
FastAPI dependency that creates per-request copies: that would reload the
DINOv3 model (1.2GB) on every request.

The API is the **sole owner** of the DINOv3 model. The Gradio UI is an HTTP
client of this API (see ``luki.app.main``) and does NOT import these
singletons — that would duplicate the model in memory.

Environment variable overrides (used by Docker):
    LUKI_QDRANT_URL   — overrides ``qdrant.url`` in config
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from luki.embeddings.dataset import load_manifest
from luki.utils.paths import config_path
from luki.embeddings.model import DinoV3Embedder
from luki.embeddings.store import QdrantStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict:
    """Load the YAML config once, with environment variable overrides."""
    with open(Path(path) if path else config_path(), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Docker networking: inside a container, localhost doesn't reach the
    # Qdrant service. The compose file sets LUKI_QDRANT_URL=http://qdrant:6334.
    if qdrant_url := os.environ.get("LUKI_QDRANT_URL"):
        cfg["qdrant"]["url"] = qdrant_url
    return cfg


@lru_cache(maxsize=1)
def get_raw_dir() -> Path:
    """Resolve the raw photos directory from config. Used to reconstruct
    full paths from the portable ``relative_path`` stored in the manifest."""
    cfg = get_config()
    return Path(cfg["data"]["raw_dir"]).resolve()


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

    Without this, the first user request would pay the 10-second model-load
    cost instead of an immediate answer. Eager warmup is the right pattern
    for user-facing ML services.
    """
    get_config()
    get_raw_dir()
    get_manifest()
    get_embedder()
    get_store()
    logger.info("All singletons initialized and ready.")
