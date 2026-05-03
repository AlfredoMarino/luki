"""Deterministic image augmentations for the augmentation-invariance eval.

Lifted from ``notebooks/03c_augmentation_task.ipynb``. Every function shares
the same signature::

    aug(img: PIL.Image, rng: np.random.Generator, **kwargs) -> PIL.Image

so that an eval runner can iterate over a dictionary of augmentations and
apply each one uniformly. The ``rng`` argument is non-optional even when
the function has no internal randomness (e.g. ``grayscale``) — the uniform
shape simplifies the runner.

Reproducibility is a hard requirement of this module: same input image +
same seeded ``rng`` must produce a byte-identical output, every time. The
unit tests in ``tests/test_augment.py`` enforce that.

Why these four:
    grayscale          — colour-channel invariance (DINOv3 trains on this)
    random_crop        — spatial invariance (DINOv3 trains heavily on this)
    brightness_jitter  — photometric invariance (DINOv3 trains on this)
    rotate             — geometric invariance OUTSIDE DINOv3's training
                          distribution (the only non-circular check)
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# A single-line type for any augmentation in the pipeline.
Augmentation = Callable[[Image.Image, np.random.Generator], Image.Image]

__all__ = [
    "Augmentation",
    "DEFAULT_AUGMENTATIONS",
    "brightness_jitter",
    "grayscale",
    "random_crop",
    "rotate",
]


def grayscale(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Convert to grayscale, then back to 3-channel RGB.

    No randomness — ``rng`` is accepted only for signature uniformity.
    Returning RGB (rather than 1-channel L) keeps the tensor shape compatible
    with vision models that expect 3 channels.
    """
    return ImageOps.grayscale(img).convert("RGB")


def random_crop(
    img: Image.Image,
    rng: np.random.Generator,
    *,
    ratio: float = 0.7,
) -> Image.Image:
    """Crop a random window covering ``ratio × ratio`` of the original area.

    ratio=0.7 drops 30% of the pixels in each dimension — aggressive enough
    to be the hardest of the four augmentations on small, mutually-similar
    cohorts (see ``03c`` cohort-density analysis), tame enough to keep the
    relevance label "the original is the only relevant doc" honest (the
    cropped image still depicts the same scene).
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    x = int(rng.integers(0, w - cw + 1))
    y = int(rng.integers(0, h - ch + 1))
    return img.crop((x, y, x + cw, y + ch))


def brightness_jitter(
    img: Image.Image,
    rng: np.random.Generator,
    *,
    factor_range: tuple[float, float] = (0.6, 1.4),
) -> Image.Image:
    """Multiply pixel intensities by a random factor in ``factor_range``.

    ±40% by default — within the typical SSL training range for ColorJitter,
    so this is in-distribution for DINOv3 (a sanity check, not a stress test).
    """
    lo, hi = factor_range
    factor = float(rng.uniform(lo, hi))
    return ImageEnhance.Brightness(img).enhance(factor)


def rotate(
    img: Image.Image,
    rng: np.random.Generator,
    *,
    angle_range: tuple[float, float] = (-30.0, 30.0),
) -> Image.Image:
    """Rotate by a random angle drawn from ``angle_range`` degrees.

    DINOv3's official augmentation pipeline includes RandomHorizontalFlip
    but **not** arbitrary rotation (verified against
    ``facebookresearch/dinov3``'s ``data/augmentations.py``). So this is the
    only one of the four augmentations that probes generalisation outside
    the training distribution. ``expand=False`` keeps the output canvas the
    same size as the input — corners get cropped, but the embedder sees a
    consistent shape.
    """
    lo, hi = angle_range
    angle = float(rng.uniform(lo, hi))
    return img.rotate(angle, expand=False)


# Default registry — what the eval runner iterates over.
DEFAULT_AUGMENTATIONS: Dict[str, Augmentation] = {
    "grayscale":  grayscale,
    "crop":       random_crop,
    "brightness": brightness_jitter,
    "rotate":     rotate,
}
