"""Smoke tests for DinoV3Embedder.

These tests actually load the model from the HuggingFace cache. If you don't
have the weights locally, the first run downloads ~1.2GB. Subsequent runs are
fast (~15s on CPU for model load, plus inference time).

Marked `slow` so you can skip them in tight loops with `pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from luki.embeddings.model import DinoV3Embedder, _resolve_device


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_falls_back_to_cpu_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _resolve_device("auto") == "cpu"


@pytest.fixture(scope="module")
def embedder():
    return DinoV3Embedder(device="cpu")


@pytest.fixture
def red_image():
    return Image.new("RGB", (300, 400), color=(220, 40, 40))


@pytest.fixture
def blue_image():
    return Image.new("RGB", (300, 400), color=(40, 40, 220))


@pytest.mark.slow
class TestEmbedder:
    def test_hidden_size(self, embedder):
        assert embedder.hidden_size == 1024

    def test_model_version_format(self, embedder):
        assert "facebook/dinov3-vitl16" in embedder.model_version
        assert "transformers-" in embedder.model_version

    def test_single_image_shape(self, embedder, red_image):
        vecs = embedder.embed([red_image])
        assert vecs.shape == (1, 1024)
        assert vecs.dtype == np.float32

    def test_batch_shape(self, embedder, red_image, blue_image):
        vecs = embedder.embed([red_image, blue_image, red_image])
        assert vecs.shape == (3, 1024)

    def test_l2_normalized(self, embedder, red_image):
        vecs = embedder.embed([red_image], normalize=True)
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5

    def test_unnormalized_has_other_norm(self, embedder, red_image):
        vecs = embedder.embed([red_image], normalize=False)
        norm = np.linalg.norm(vecs[0])
        # Raw DINOv3 pooler_output is not unit-norm in general
        assert abs(norm - 1.0) > 1e-3

    def test_determinism(self, embedder, red_image):
        a = embedder.embed([red_image])
        b = embedder.embed([red_image])
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    def test_empty_input(self, embedder):
        vecs = embedder.embed([])
        assert vecs.shape == (0, 1024)

    def test_different_images_different_embeddings(self, embedder, red_image, blue_image):
        vecs = embedder.embed([red_image, blue_image])
        cos_sim = float(np.dot(vecs[0], vecs[1]))
        # They won't be orthogonal (both are solid colors, similar statistics),
        # but definitely not identical.
        assert cos_sim < 0.999
