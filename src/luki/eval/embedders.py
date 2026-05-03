"""Vision encoder wrappers for the model-comparison eval.

Lifted from ``notebooks/03e_model_comparison.ipynb``. Three encoders behind
one signature, so the runner can iterate models the same way it iterates
augmentations.

Common contract::

    class XEmbedder(BaseEmbedder):
        dim: int
        model_version: str
        def embed(self, images: list[PIL.Image]) -> np.ndarray: ...

The output is **always L2-normalised** (so downstream cosine similarity is
just a dot product). All three models output projection embeddings via
``get_image_features().pooler_output`` in transformers ≥ 5; we normalise
explicitly so callers don't have to remember.

Why three encoders, not just DINOv3:
    SSL (DINOv3) and contrastive image-text (CLIP / SigLIP) imprint
    different inductive biases on the embedding space. The model-comparison
    eval (``03e``) is the only thing that lets us defend the choice of
    DINOv3 in the LUKI README — without these wrappers there's no
    apples-to-apples comparison.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image

__all__ = [
    "BaseEmbedder",
    "ClipEmbedder",
    "DinoEmbedder",
    "SiglipEmbedder",
]


@runtime_checkable
class BaseEmbedder(Protocol):
    """Common interface every eval embedder must satisfy.

    Why a Protocol and not an abstract base class:
        - DinoEmbedder wraps an existing class without inheriting from us
          (we don't want eval/ to depend on a base in eval/embedders/).
        - Duck-typing is fine here; protocols give type-checker support
          without runtime overhead.
    """

    dim: int
    model_version: str

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Return an ``(N, self.dim)`` L2-normalised float32 array."""
        ...


# --------------------------------------------------------------------------- #
# DINOv3 — wraps the production embedder so we don't duplicate model loading. #
# --------------------------------------------------------------------------- #


class DinoEmbedder:
    """Adapter around the production ``DinoV3Embedder``.

    We deliberately do not inherit from it: this module is for *eval*, not
    *production*. The adapter pattern keeps the boundary clean — if the
    production embedder's signature changes, only this wrapper updates.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str = "auto",
    ):
        # Imported lazily so this module loads even if torch is unhappy.
        from luki.embeddings.model import DinoV3Embedder

        self._inner = DinoV3Embedder(model_name=model_name, device=device)
        self.dim = 1024
        self.model_version = self._inner.model_version

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        out = self._inner.embed(images)
        # The production embedder already L2-normalises (see config.normalize),
        # but we re-normalise here to make the contract explicit for any
        # future configuration change.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(norms, 1e-12, None)).astype(np.float32)


# --------------------------------------------------------------------------- #
# CLIP — openai/clip-vit-base-patch32, image branch only.                     #
# --------------------------------------------------------------------------- #


class ClipEmbedder:
    """OpenAI CLIP-ViT-B/32 image embeddings (512-dim).

    We use ``CLIPImageProcessor`` (image-only) rather than the full
    ``CLIPProcessor`` because the latter eagerly loads a text tokenizer we
    never use — pointless coupling for an image-only path.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 8,
    ):
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = 512
        self.model_version = model_name

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs).pooler_output
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# SigLIP — google/siglip-base-patch16-224, image branch only.                 #
# --------------------------------------------------------------------------- #


class SiglipEmbedder:
    """Google SigLIP base/16 image embeddings (768-dim).

    Sigmoid-loss successor to CLIP — different training objective, similar
    output geometry. Same image-only-processor pattern as ``ClipEmbedder``,
    additionally avoiding a SentencePiece dependency that the full
    ``SiglipProcessor`` pulls in for its text tokenizer.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "auto",
        batch_size: int = 8,
    ):
        from transformers import SiglipImageProcessor, SiglipModel

        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = 768
        self.model_version = model_name

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs).pooler_output
                feats = torch.nn.functional.normalize(feats, dim=-1)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _resolve_device(device: str) -> str:
    """``"auto"`` → ``cuda`` if available, else ``cpu``. Otherwise pass-through."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
