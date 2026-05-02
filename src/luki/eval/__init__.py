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

from luki.eval.metrics import (
    average_precision,
    bootstrap_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "average_precision",
    "bootstrap_ci",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
