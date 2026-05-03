"""Embedding-space diagnostics for hypothesis-distinguishing experiments.

These are NOT metrics. They are descriptive lenses on the embedding space
itself — used to *explain* metric outcomes, not to score the model.
Productivized from ``notebooks/03c_augmentation_task.ipynb`` after the
H1-vs-H2 hypothesis-test moment that gave the module its design.

The senior-track use case: when two hypotheses both predict the same
headline metric, you need *secondary predictions* to distinguish them.
The functions below produce those secondary measurements.

    intra_cohort_similarity  — T1: per-cohort mean pairwise cosine sim
    nearest_neighbor_similarity — T3: per-photo 1-NN cosine sim, by cohort

T2 (failure forensics) is not a function but a *pattern* — it's a join
between the per-query result table (which records the top-1 winner) and
the manifest. Codified as a helper here for reuse across eval tasks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "annotate_failure_forensics",
    "intra_cohort_similarity",
    "nearest_neighbor_similarity",
]


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Assumes inputs are L2-normalized.

    For unit-norm vectors the dot product equals the cosine, so this is
    just ``vecs @ vecs.T``. We blank the diagonal so self-similarity (= 1.0
    by construction) doesn't pollute the per-row aggregates.
    """
    if vecs.ndim != 2:
        raise ValueError(f"vecs must be 2-D, got shape {vecs.shape}")
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, np.nan)
    return sim


def intra_cohort_similarity(
    vecs: np.ndarray,
    cohorts: Sequence,
) -> pd.DataFrame:
    """T1 — mean pairwise cosine similarity *within* each cohort.

    Higher = the cohort is denser in embedding space, i.e. its members are
    more confusable for one another. In LUKI's case digital scored ~0.27
    while every film roll scored 0.07–0.13 — that 2-4× density gap is what
    makes digital photos so vulnerable to crop perturbation.

    Returns
    -------
    pd.DataFrame
        Columns: ``cohort``, ``n``, ``mean_intra_cos``. Sorted descending
        by similarity (densest cohorts first).
    """
    if len(cohorts) != len(vecs):
        raise ValueError(
            f"cohorts ({len(cohorts)}) and vecs ({len(vecs)}) length mismatch"
        )
    sim = _cosine_matrix(vecs)
    df = pd.DataFrame({"cohort": list(cohorts)})
    rows = []
    for c, idx in df.groupby("cohort", dropna=False).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            # Single-member cohort: no pairs → similarity undefined.
            rows.append({"cohort": c, "n": len(idx), "mean_intra_cos": float("nan")})
            continue
        sub = sim[np.ix_(idx, idx)]
        rows.append({"cohort": c, "n": len(idx), "mean_intra_cos": float(np.nanmean(sub))})
    return (
        pd.DataFrame(rows)
        .sort_values("mean_intra_cos", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def nearest_neighbor_similarity(
    vecs: np.ndarray,
    cohorts: Sequence,
) -> pd.DataFrame:
    """T3 — per-photo 1-NN cosine similarity, returned with cohort labels.

    This is the lens that distinguishes "uniformly dense cohort" from
    "sparse cohort with near-duplicate clusters." If the cohort *mean*
    intra-cohort sim is moderate but the per-photo *1-NN* sim is very high,
    the cohort isn't a uniform blob — it has tight pairs/triplets. In LUKI
    the digital cohort showed exactly that: mean ~0.27, median 1-NN ~0.91.

    Returns
    -------
    pd.DataFrame
        One row per photo, columns: ``cohort``, ``nn_sim``. Use ``groupby``
        for per-cohort aggregates (mean / median / boxplot etc.).
    """
    if len(cohorts) != len(vecs):
        raise ValueError(
            f"cohorts ({len(cohorts)}) and vecs ({len(vecs)}) length mismatch"
        )
    sim = _cosine_matrix(vecs)
    nn = np.nanmax(sim, axis=1)
    return pd.DataFrame({"cohort": list(cohorts), "nn_sim": nn})


def annotate_failure_forensics(
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    cohort_col: str = "roll_tags",
    fallback_col: str = "medium",
) -> pd.DataFrame:
    """T2 — annotate per-query results with the cohort of the top-1 winner.

    Adds two columns:
        ``query_cohort``  — cohort of the original query
        ``winner_cohort`` — cohort of whoever won top-1 (could be the
                            query itself on a successful query)
        ``same_cohort``   — convenience boolean

    Cohort is resolved as ``cohort_col`` if non-null, else ``fallback_col``
    — same logic the notebooks use to fold the digital cohort
    (``roll_tags == NaN``) into the same axis as the film rolls.

    The ``results`` DataFrame must contain ``top1_path`` (the relative_path
    of the top-1 hit) — produced by an eval runner that records forensic
    columns on each query.
    """
    required = {"top1_path"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {missing}")

    path_to_cohort = dict(zip(manifest["relative_path"], manifest[cohort_col]))
    path_to_fallback = dict(zip(manifest["relative_path"], manifest[fallback_col]))

    def _cohort_of(path: str | None, query_row: pd.Series | None = None) -> str | None:
        if path is None:
            return None
        c = path_to_cohort.get(path)
        if c is None or (isinstance(c, float) and pd.isna(c)):
            return path_to_fallback.get(path)
        return c

    out = results.copy()
    # Query cohort — prefer the result row's own metadata if present, else
    # look up by the query's relative_path / filename. We expect the runner
    # to carry the manifest columns through (see queries.build_*_task).
    if cohort_col in out.columns:
        out["query_cohort"] = [
            r[cohort_col] if pd.notna(r[cohort_col]) else r.get(fallback_col)
            for _, r in out.iterrows()
        ]
    else:
        # Fallback: look up by filename
        if "filename" not in out.columns:
            raise ValueError(
                f"results lacks both {cohort_col!r} and 'filename' columns"
            )
        path_by_filename = dict(zip(manifest["filename"], manifest["relative_path"]))
        out["query_cohort"] = [
            _cohort_of(path_by_filename.get(fn)) for fn in out["filename"]
        ]

    out["winner_cohort"] = [_cohort_of(p) for p in out["top1_path"]]
    out["same_cohort"] = out["query_cohort"] == out["winner_cohort"]
    return out
