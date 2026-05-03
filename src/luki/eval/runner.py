"""Generic eval runner — corpus embed + brute-force cosine + per-query metrics.

Lifted from ``notebooks/03d_session_task.ipynb``. Two design choices to defend:

**No Qdrant.** The full embedding matrix is pulled into memory once and
reused. Brute-force cosine over a 200-photo corpus is microseconds per
query (one matmul); a Qdrant round-trip would be ~ms per query. Memory
cost: a 200×1024 float32 matrix is 800 KB. Trivial. The same runner will
serve ``03e``'s model comparison without re-indexing Qdrant per model.

**Generic on `group_col`.** Both the *roll* hold-out (LUKI's data) and a
hypothetical *session* hold-out are the same code with a different
argument. The runner doesn't know about LUKI's specific schema.

Output contract: one row per query, with metric columns plus
forensic columns (``top1_path``, ``top1_hash``) for downstream
``annotate_failure_forensics``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from luki.eval.metrics import (
    average_precision,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = ["evaluate_holdout"]


def evaluate_holdout(
    eval_set: pd.DataFrame,
    embeddings: np.ndarray,
    corpus_paths: Sequence[str],
    corpus_manifest: pd.DataFrame,
    group_col: str = "roll_tags",
    top_k: int = 50,
) -> pd.DataFrame:
    """Leave-one-out evaluation against a fully in-memory embedding matrix.

    For each query in ``eval_set``:
        1. Look up its row in the embedding matrix via its relative_path.
        2. Compute cosine similarity to every other row.
        3. Sort by descending similarity, take top_k.
        4. Build the binary relevance vector by checking each hit's
           ``group_col`` value (joined from ``corpus_manifest``).
        5. Compute Recall@10/@20, P@10, RR, AP.

    Parameters
    ----------
    eval_set
        DataFrame of query rows. Must include ``relative_path``,
        ``filename``, ``file_hash``, and ``group_col``.
    embeddings
        ``(N, D)`` matrix, **must be L2-normalized** (cosine == dot product).
        Order corresponds to ``corpus_paths``.
    corpus_paths
        Length-``N`` list of relative paths matching ``embeddings`` rows.
        Used to align hits with the manifest's ``group_col`` values.
    corpus_manifest
        The full manifest for all corpus photos (not just the eval set);
        provides the relevance labels via ``relative_path → group_col``.
    group_col
        Column name to treat as the relevance proxy.
    top_k
        How many hits to retrieve and score. Recall@10 etc. are still
        computed at the smaller cuts.

    Returns
    -------
    pd.DataFrame
        One row per query. Columns:
            filename, group, n_relevant,
            Recall@10, Recall@20, P@10, RR, AP,
            top1_path, top1_hash    (forensic, for failure analysis)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if len(embeddings) != len(corpus_paths):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) and corpus_paths "
            f"({len(corpus_paths)}) must match"
        )
    required = {"relative_path", "filename", "file_hash", group_col}
    missing = required - set(eval_set.columns)
    if missing:
        raise ValueError(f"eval_set is missing required columns: {missing}")

    path_to_idx = {p: i for i, p in enumerate(corpus_paths)}
    path_to_group = dict(zip(corpus_manifest["relative_path"], corpus_manifest[group_col]))
    path_to_hash = dict(zip(corpus_manifest["relative_path"], corpus_manifest["file_hash"]))

    rows = []
    for _, q in eval_set.iterrows():
        try:
            q_idx = path_to_idx[q["relative_path"]]
        except KeyError as e:
            raise ValueError(
                f"query {q['relative_path']!r} not found in corpus_paths — "
                f"eval_set must be a subset of the embedded corpus"
            ) from e
        q_group = q[group_col]

        # Cosine similarities (L2-normed → dot product). Self-mask before sorting.
        sims = embeddings @ embeddings[q_idx]
        sims[q_idx] = -np.inf
        order = np.argsort(-sims)[:top_k]
        top_paths = [corpus_paths[i] for i in order]

        # Binary relevance: same group_col value → relevant
        relevance = np.array(
            [int(path_to_group.get(p) == q_group) for p in top_paths],
            dtype=int,
        )

        # Total relevants in the corpus excluding self
        n_relevant = sum(
            1 for v in corpus_manifest[group_col]
            if v == q_group
        ) - 1
        if n_relevant < 0:
            n_relevant = 0  # shouldn't happen if eval_set was built from manifest

        rows.append({
            "filename":   q["filename"],
            "group":      q_group,
            "n_relevant": n_relevant,
            "Recall@10":  recall_at_k(relevance, 10, n_relevant),
            "Recall@20":  recall_at_k(relevance, 20, n_relevant),
            "P@10":       precision_at_k(relevance, 10),
            "RR":         reciprocal_rank(relevance),
            "AP":         average_precision(relevance, n_relevant),
            "top1_path":  top_paths[0] if top_paths else None,
            "top1_hash":  path_to_hash.get(top_paths[0]) if top_paths else None,
        })

    return pd.DataFrame(rows)
