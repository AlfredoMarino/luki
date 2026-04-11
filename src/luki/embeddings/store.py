"""Qdrant wrapper for the LUKI embeddings pipeline.

Responsibilities:
- Create / verify the collection (idempotent).
- Create payload indexes for fast hybrid filtering.
- Upsert vectors with photo metadata as payload.
- Retrieve existing points by MD5 to support caching.
- Run similarity search with optional metadata filters.

The class does not know anything about DINOv3, PIL, or the manifest schema —
callers pass already-computed vectors + arbitrary payload dicts. Keeping the
store model-agnostic lets us swap the embedder (or the store backend) later.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


# Payload fields we want to filter by. Keyword indexes give O(log n) filtered
# search instead of the default linear scan during HNSW traversal.
_KEYWORD_INDEX_FIELDS: tuple[str, ...] = (
    "medium",
    "camera",
    "session_name",
    "model_version",
)
_INTEGER_INDEX_FIELDS: tuple[str, ...] = ("year",)


def md5_to_uuid(md5_hex: str) -> str:
    """Convert a 32-char MD5 hex digest into a deterministic UUID string.

    Qdrant only accepts int or UUID as point IDs. Using a hash-derived UUID
    makes upserts idempotent: re-ingesting the same photo overwrites the same
    point instead of creating duplicates.
    """
    if len(md5_hex) != 32:
        raise ValueError(f"Expected 32-char MD5 hex, got {len(md5_hex)}: {md5_hex!r}")
    return str(uuid.UUID(hex=md5_hex))


class QdrantStore:
    """Thin, opinionated wrapper over QdrantClient for LUKI."""

    def __init__(
        self,
        url: str | None = None,
        collection_name: str = "luki_photos",
        vector_size: int = 1024,
        distance: str = "Cosine",
        *,
        location: str | None = None,
        prefer_grpc: bool = True,
    ) -> None:
        """Create the client and ensure the collection exists.

        Use ``url`` for a running Qdrant server (Docker / Cloud), or
        ``location=":memory:"`` for an in-process client — ideal for tests.
        """
        if location is not None:
            self.client = QdrantClient(location=location)
        else:
            if url is None:
                raise ValueError("Must provide either `url` or `location`.")
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc)

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Collection setup                                                    #
    # ------------------------------------------------------------------ #
    def _ensure_collection(self) -> None:
        """Create the collection + payload indexes if they don't exist."""
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists with "
                    f"vector_size={existing_size}, but caller requested "
                    f"{self.vector_size}. Delete the collection or use a new name."
                )
            logger.info("Collection '%s' already exists.", self.collection_name)
            return

        logger.info(
            "Creating collection '%s' (size=%d, distance=%s)",
            self.collection_name,
            self.vector_size,
            self.distance,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=qmodels.Distance[self.distance.upper()],
            ),
        )

        for field in _KEYWORD_INDEX_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        for field in _INTEGER_INDEX_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #
    def upsert_batch(
        self,
        file_hashes: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert a batch of points. vectors must be shape (N, vector_size)."""
        if vectors.ndim != 2 or vectors.shape[1] != self.vector_size:
            raise ValueError(
                f"vectors must have shape (N, {self.vector_size}), got {vectors.shape}"
            )
        if not (len(file_hashes) == vectors.shape[0] == len(payloads)):
            raise ValueError("file_hashes, vectors, and payloads must have same length")

        points = [
            qmodels.PointStruct(
                id=md5_to_uuid(h),
                vector=v.tolist(),
                payload=p,
            )
            for h, v, p in zip(file_hashes, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    # ------------------------------------------------------------------ #
    # Cache lookups                                                       #
    # ------------------------------------------------------------------ #
    def existing_hashes_for_version(
        self,
        file_hashes: Iterable[str],
        model_version: str,
    ) -> set[str]:
        """Return the subset of file_hashes already stored with this model_version.

        Anything stored with a *different* model_version is treated as absent,
        so it gets re-embedded with the current model. This is our cache
        invalidation strategy.
        """
        hashes = list(file_hashes)
        if not hashes:
            return set()

        ids = [md5_to_uuid(h) for h in hashes]
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=["file_hash", "model_version"],
            with_vectors=False,
        )
        return {
            r.payload["file_hash"]
            for r in records
            if r.payload and r.payload.get("model_version") == model_version
        }

    # ------------------------------------------------------------------ #
    # Search                                                              #
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        query_filter: qmodels.Filter | None = None,
    ) -> list[qmodels.ScoredPoint]:
        """k-NN search with optional metadata filter."""
        if query_vector.ndim != 1 or query_vector.shape[0] != self.vector_size:
            raise ValueError(
                f"query_vector must be shape ({self.vector_size},), got {query_vector.shape}"
            )
        # `query_points` is the modern API (qdrant-client >= 1.10).
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.points

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count
