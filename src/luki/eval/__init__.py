"""Evaluation utilities for LUKI's embedding-based retrieval.

Subpackage layout (built incrementally across Phase 1.5):
    metrics.py    — IR metrics (Recall@k, MRR, AP) + bootstrap CI
    augment.py    — deterministic PIL augmentations  (03c)
    queries.py    — eval-set builders                 (03c, 03d)
    runner.py     — corpus embed + cosine rank        (03d)
    embedders.py  — DINOv3 / CLIP / SigLIP wrappers   (03e)
    cli.py        — `luki-eval` entry point           (03e)

Eval supervision is intentionally decoupled from the production Qdrant index:
relevance labels are looked up in the manifest (the source of truth), not in
the vector store's payload.
"""

from luki.eval.augment import (
    DEFAULT_AUGMENTATIONS,
    Augmentation,
    brightness_jitter,
    grayscale,
    random_crop,
    rotate,
)
from luki.eval.diagnostics import (
    annotate_failure_forensics,
    intra_cohort_similarity,
    nearest_neighbor_similarity,
)
from luki.eval.metrics import (
    average_precision,
    bootstrap_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from luki.eval.queries import (
    build_augmentation_task,
    build_holdout_task,
    stratified_sample,
)
from luki.eval.runner import evaluate_holdout
from luki.eval.embedders import (
    BaseEmbedder,
    ClipEmbedder,
    DinoEmbedder,
    SiglipEmbedder,
)

__all__ = [
    # metrics
    "average_precision",
    "bootstrap_ci",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    # augmentations
    "Augmentation",
    "DEFAULT_AUGMENTATIONS",
    "brightness_jitter",
    "grayscale",
    "random_crop",
    "rotate",
    # query-set builders
    "build_augmentation_task",
    "build_holdout_task",
    "stratified_sample",
    # runner
    "evaluate_holdout",
    # diagnostics
    "annotate_failure_forensics",
    "intra_cohort_similarity",
    "nearest_neighbor_similarity",
    # embedders
    "BaseEmbedder",
    "ClipEmbedder",
    "DinoEmbedder",
    "SiglipEmbedder",
]
