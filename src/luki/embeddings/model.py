"""DINOv3 image embedder used by the LUKI embeddings pipeline.

Why a class and not a function?
- We want to load the model ONCE and reuse it across many batches. A function
  would reload weights on every call (or rely on globals, which is worse).
- It's the natural place to pin configuration: device, dtype, model version.
- It makes unit testing and swapping backbones (DINOv2, CLIP, SigLIP, ...)
  trivial — we just write another class with the same `embed()` signature.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' into the best available device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class DinoV3Embedder:
    """Loads a DINOv3 model and turns PIL images into L2-normalized embeddings.

    The public API is just `embed(images) -> np.ndarray` so callers never
    have to touch PyTorch, tokens, or device management.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)

        logger.info("Loading DINOv3 processor: %s", model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        logger.info("Loading DINOv3 model: %s (device=%s)", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        self.hidden_size: int = self.model.config.hidden_size
        # Build a version string combining model + transformers to support
        # cache invalidation in the vector store.
        self.model_version: str = f"{model_name}@transformers-{transformers.__version__}"

        logger.info(
            "Embedder ready — hidden_size=%d, version=%s",
            self.hidden_size,
            self.model_version,
        )

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def embed(
        self,
        images: Sequence[Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        """Run a forward pass and return shape (N, hidden_size) as float32 numpy.

        We use `pooler_output` as the global image embedding (equivalent to
        the CLS token after the model's projection head) — it is the standard
        contract for image-level features in HuggingFace.

        `@torch.inference_mode()` is strictly stronger than `torch.no_grad()`:
        it also disables the autograd version counter, giving ~5-10% speedup
        and a clear "this is inference" signal.
        """
        if not images:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        vecs = outputs.pooler_output  # (N, hidden_size)

        if normalize:
            # L2-normalize each row so dot product == cosine similarity.
            vecs = F.normalize(vecs, p=2, dim=-1)

        return vecs.detach().cpu().numpy().astype(np.float32, copy=False)
