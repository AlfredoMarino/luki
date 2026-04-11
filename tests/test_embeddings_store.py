"""Tests for QdrantStore using an in-memory Qdrant client.

No Docker needed — QdrantClient(":memory:") runs the full engine in-process.
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client.http import models as qmodels

from luki.embeddings.store import QdrantStore, md5_to_uuid


# --------------------------------------------------------------------------- #
# md5_to_uuid                                                                  #
# --------------------------------------------------------------------------- #
class TestMd5ToUuid:
    def test_deterministic(self):
        md5 = "a" * 32
        assert md5_to_uuid(md5) == md5_to_uuid(md5)

    def test_different_inputs_different_outputs(self):
        assert md5_to_uuid("a" * 32) != md5_to_uuid("b" * 32)

    def test_real_md5_format(self):
        md5 = "d41d8cd98f00b204e9800998ecf8427e"
        result = md5_to_uuid(md5)
        assert result == "d41d8cd9-8f00-b204-e980-0998ecf8427e"

    def test_rejects_short(self):
        with pytest.raises(ValueError):
            md5_to_uuid("abc")

    def test_rejects_long(self):
        with pytest.raises(ValueError):
            md5_to_uuid("a" * 33)


# --------------------------------------------------------------------------- #
# QdrantStore                                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture
def store():
    """Fresh in-memory store with 4-dim vectors (cheap for tests)."""
    return QdrantStore(
        location=":memory:",
        collection_name="test_luki",
        vector_size=4,
        distance="Cosine",
    )


@pytest.fixture
def sample_hashes():
    return [
        "a" * 32,
        "b" * 32,
        "c" * 32,
    ]


@pytest.fixture
def sample_vectors():
    # Already L2-normalized for cosine sanity
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_payloads():
    return [
        {
            "file_hash": "a" * 32,
            "medium": "digital",
            "camera": "canon-500d",
            "year": 2026,
            "session_name": "20260201_chile",
            "model_version": "test@1.0",
        },
        {
            "file_hash": "b" * 32,
            "medium": "film",
            "camera": "pentax-k1000",
            "year": 2025,
            "session_name": "20250810_madrid",
            "model_version": "test@1.0",
        },
        {
            "file_hash": "c" * 32,
            "medium": "digital",
            "camera": "canon-500d",
            "year": 2026,
            "session_name": "20260201_chile",
            "model_version": "test@1.0",
        },
    ]


class TestCollectionSetup:
    def test_creates_collection(self, store):
        assert store.client.collection_exists("test_luki")
        assert store.count() == 0

    def test_idempotent_creation(self, store):
        # Re-creating the store against the same client should not blow up.
        again = QdrantStore(
            location=":memory:",   # new client, but test the ensure logic
            collection_name="test_luki",
            vector_size=4,
        )
        assert again.client.collection_exists("test_luki")

    def test_rejects_wrong_vector_size(self, store):
        # Reuse the same in-memory client: the collection already exists with
        # size=4. Asking for size=8 must raise.
        second = QdrantStore.__new__(QdrantStore)
        second.client = store.client
        second.collection_name = store.collection_name
        second.vector_size = 8  # mismatched
        second.distance = store.distance
        with pytest.raises(ValueError, match="vector_size"):
            second._ensure_collection()


class TestUpsertAndSearch:
    def test_upsert_then_count(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        assert store.count() == 3

    def test_search_returns_nearest(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].payload["file_hash"] == "a" * 32

    def test_upsert_is_idempotent(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        # Same IDs -> no duplication
        assert store.count() == 3

    def test_search_with_filter(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        film_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="medium", match=qmodels.MatchValue(value="film"))]
        )
        results = store.search(query, top_k=5, query_filter=film_filter)
        assert len(results) == 1
        assert results[0].payload["medium"] == "film"
        assert results[0].payload["file_hash"] == "b" * 32

    def test_rejects_wrong_vector_dim(self, store, sample_hashes, sample_payloads):
        bad = np.zeros((3, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            store.upsert_batch(sample_hashes, bad, sample_payloads)


class TestCacheLookup:
    def test_existing_hashes_for_version(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        present = store.existing_hashes_for_version(sample_hashes, model_version="test@1.0")
        assert present == set(sample_hashes)

    def test_different_version_treated_as_absent(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes, sample_vectors, sample_payloads)
        present = store.existing_hashes_for_version(sample_hashes, model_version="other@2.0")
        assert present == set()

    def test_partial_cache_hit(self, store, sample_hashes, sample_vectors, sample_payloads):
        store.upsert_batch(sample_hashes[:2], sample_vectors[:2], sample_payloads[:2])
        present = store.existing_hashes_for_version(sample_hashes, model_version="test@1.0")
        assert present == {"a" * 32, "b" * 32}

    def test_empty_input(self, store):
        assert store.existing_hashes_for_version([], model_version="test@1.0") == set()
