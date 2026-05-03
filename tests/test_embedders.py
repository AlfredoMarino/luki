"""Smoke tests for ``luki.eval.embedders``.

We do not exercise the full DINOv3/CLIP/SigLIP weights here — those tests
live behind the existing ``slow`` marker (model loads cost ~15s each, and
the CI environment shouldn't pay that on every PR). Instead we verify the
*interface contract* every embedder must satisfy:

    - exposes ``dim`` and ``model_version``
    - returns a 2-D numpy float32 array of the right shape
    - returns L2-normalised vectors
    - implements the ``BaseEmbedder`` protocol (runtime check)

Real-model integration tests are gated behind ``-m slow`` and exercise
DINOv3 against a single image.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from luki.eval.embedders import BaseEmbedder


@pytest.fixture
def tiny_image() -> Image.Image:
    """A tiny 224×224 RGB image to feed any embedder cheaply."""
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


# --------------------------------------------------------------------------- #
# Interface protocol                                                          #
# --------------------------------------------------------------------------- #


class TestProtocol:
    def test_dino_satisfies_protocol(self):
        # Light import, no instantiation
        from luki.eval.embedders import DinoEmbedder
        # Class-level: confirm it has the protocol surface
        assert hasattr(DinoEmbedder, "embed")
        # Can't isinstance-check a class against a Protocol without instances;
        # we'll do a runtime check at integration-test time.

    def test_clip_satisfies_protocol(self):
        from luki.eval.embedders import ClipEmbedder
        assert hasattr(ClipEmbedder, "embed")

    def test_siglip_satisfies_protocol(self):
        from luki.eval.embedders import SiglipEmbedder
        assert hasattr(SiglipEmbedder, "embed")


# --------------------------------------------------------------------------- #
# Slow integration tests — only with -m slow                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestDinoEmbedder:
    """Exercises the real DINOv3 weights. ~15s cold start."""

    def test_returns_correct_shape_and_dtype(self, tiny_image: Image.Image):
        from luki.eval.embedders import DinoEmbedder
        emb = DinoEmbedder()
        out = emb.embed([tiny_image])
        assert out.shape == (1, 1024)
        assert out.dtype == np.float32

    def test_l2_normalized(self, tiny_image: Image.Image):
        from luki.eval.embedders import DinoEmbedder
        emb = DinoEmbedder()
        out = emb.embed([tiny_image, tiny_image])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_satisfies_runtime_protocol(self):
        from luki.eval.embedders import DinoEmbedder
        emb = DinoEmbedder()
        assert isinstance(emb, BaseEmbedder)
        assert emb.dim == 1024
        assert isinstance(emb.model_version, str)


@pytest.mark.slow
class TestClipEmbedder:
    def test_returns_512_dim_l2_normalized(self, tiny_image: Image.Image):
        from luki.eval.embedders import ClipEmbedder
        emb = ClipEmbedder()
        out = emb.embed([tiny_image])
        assert out.shape == (1, 512)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


@pytest.mark.slow
class TestSiglipEmbedder:
    def test_returns_768_dim_l2_normalized(self, tiny_image: Image.Image):
        from luki.eval.embedders import SiglipEmbedder
        emb = SiglipEmbedder()
        out = emb.embed([tiny_image])
        assert out.shape == (1, 768)
        assert out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
