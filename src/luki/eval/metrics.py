"""Information-retrieval metrics, derived in ``notebooks/03b_ir_metrics.ipynb``.

All metrics share one input contract:

    relevance:  1-D binary array (0/1) listing the relevance label of each
                returned doc, **in ranked order** (rank 1 first).
    n_relevant: total number of relevant docs in the corpus, including any
                that were *not* in the returned list. Required by Recall@k
                and AP because both penalise unretrieved relevants.

This contract is intentionally narrower than passing dataframes around: every
metric is a pure function over a numpy array, easy to unit-test with
hand-crafted rank lists. See ``tests/test_metrics.py``.

Why bootstrap_ci lives here too: every reported metric should come with a CI,
so the helper that computes it belongs next to the metrics that need it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "average_precision",
    "bootstrap_ci",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def _as_binary_array(relevance: Iterable[int] | np.ndarray) -> np.ndarray:
    """Coerce to a 1-D int array and validate values are in {0, 1}."""
    arr = np.asarray(relevance, dtype=int)
    if arr.ndim != 1:
        raise ValueError(f"relevance must be 1-D, got shape {arr.shape}")
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError("relevance must contain only 0/1 entries")
    return arr


def precision_at_k(
    relevance: Iterable[int] | np.ndarray,
    k: int,
) -> float:
    """Fraction of the top-k that are relevant.

    Reads as: *of what I displayed on the page, how much was on-topic?* —
    the page-quality metric. Note the asymmetry with ``recall_at_k``: this
    is divided by ``k`` (the number of slots shown), recall is divided by
    ``n_relevant`` (the size of the relevant set in the corpus).

    **Ceiling-effect caveat.** ``precision_at_k`` has a max of
    ``min(1.0, n_relevant / k)``. When ``n_relevant < k`` a *perfect* model
    cannot score 1.0 — there aren't enough relevants in the corpus to fill
    the page. Averaging precision_at_k across heterogeneous queries (some
    with many relevants, some with few) silently rewards large-relevant-set
    queries. Always report alongside recall_at_k, or use R-Precision (set
    k = n_relevant per query) to remove the ceiling.

    >>> precision_at_k([1, 0, 1, 0, 0], k=3)
    0.6666666666666666
    >>> precision_at_k([0, 0, 0, 0, 0], k=5)
    0.0
    """
    rel = _as_binary_array(relevance)
    if k <= 0:
        return 0.0
    # Cap k at list length — asking for top-k when fewer were returned just
    # evaluates over what we have. (Prevents division surprises if the
    # caller passes k larger than the retrieved list.)
    effective_k = min(k, len(rel))
    if effective_k == 0:
        return 0.0
    return float(rel[:effective_k].sum() / effective_k)


def recall_at_k(
    relevance: Iterable[int] | np.ndarray,
    k: int,
    n_relevant: int,
) -> float:
    """Fraction of all corpus-relevant docs that landed in the top-k.

    Returns ``nan`` when ``n_relevant == 0``: a query with no relevant docs
    has no defined recall, and silently substituting 0 would bias an average
    downward. See the digital-cohort discussion in ``03a``.

    >>> recall_at_k([1, 0, 1, 0, 0], k=3, n_relevant=4)
    0.5
    >>> recall_at_k([0, 0, 0, 0, 0], k=5, n_relevant=3)
    0.0
    """
    rel = _as_binary_array(relevance)
    if n_relevant < 0:
        raise ValueError("n_relevant must be non-negative")
    if n_relevant == 0:
        return float("nan")
    if k <= 0:
        return 0.0
    return float(rel[:k].sum() / n_relevant)


def reciprocal_rank(relevance: Iterable[int] | np.ndarray) -> float:
    """1 / rank of the first relevant doc, or 0 if none in the list.

    By convention RR=0 when the returned list contains no relevant doc — this
    matches every IR textbook (Manning et al., 2008) and ``trec_eval``.

    >>> reciprocal_rank([0, 0, 1, 0, 1])
    0.3333333333333333
    >>> reciprocal_rank([0, 0, 0])
    0.0
    """
    rel = _as_binary_array(relevance)
    hits = np.where(rel == 1)[0]
    if len(hits) == 0:
        return 0.0
    return float(1.0 / (hits[0] + 1))  # +1 → 1-indexed rank


def average_precision(
    relevance: Iterable[int] | np.ndarray,
    n_relevant: int,
) -> float:
    """Standard binary-relevance Average Precision.

    AP = (1 / n_relevant) * sum_{k=1..N} P@k * rel_k
    where N = len(relevance) and P@k = (#relevants in top-k) / k.

    Dividing by the *corpus* n_relevant (not by the number of retrieved
    relevants) is what makes AP penalise unretrieved relevants — a missing
    doc contributes 0 to the sum.

    Returns ``nan`` when ``n_relevant == 0`` (same rationale as ``recall_at_k``).

    >>> # 2 relevants found at ranks 1 and 3, total 2 in corpus
    >>> ap = average_precision([1, 0, 1, 0], n_relevant=2)
    >>> round(ap, 3)
    0.833
    """
    rel = _as_binary_array(relevance)
    if n_relevant < 0:
        raise ValueError("n_relevant must be non-negative")
    if n_relevant == 0:
        return float("nan")
    if rel.size == 0:
        return 0.0
    cum_hits = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1)
    precisions = cum_hits / ranks
    return float((precisions * rel).sum() / n_relevant)


def bootstrap_ci(
    values: Iterable[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the **mean** of ``values``.

    Returns ``(mean, lo, hi)`` where (lo, hi) is the (alpha/2, 1-alpha/2)
    quantile interval over ``n_boot`` bootstrap resamples (with replacement)
    of the mean.

    Why bootstrap, not mean ± std:
        - Per-query metrics are bounded in [0, 1] and frequently skewed; the
          Gaussian assumption behind ±std is wrong on the tails.
        - With small N the std is itself a noisy estimator; the bootstrap CI
          is wider but honest about that fact.
        - Generalises trivially to medians, ratios, or any other estimator.

    NaN values are dropped before resampling — relevant for queries that
    returned nan from ``recall_at_k`` / ``average_precision`` because they
    had no corpus-relevant docs.

    >>> rng_seed = 42
    >>> mean, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], seed=rng_seed)
    >>> round(mean, 2), lo < mean < hi
    (0.3, True)
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = len(arr)
    # Vectorised bootstrap — draw all resamples at once instead of looping
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lo), float(hi)
