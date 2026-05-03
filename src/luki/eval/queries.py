"""Eval-set builders.

A *query set* is a DataFrame whose rows describe individual queries. Each
eval task (augmentation, session, ...) builds its own query set with this
shape, then the runner consumes it uniformly. Keeping the data layout
consistent across tasks is what lets the runner stay generic.

Currently provides:
    stratified_sample       — generic helper
    build_augmentation_task — query set for the augmentation-invariance eval

Future tasks (``03d`` onwards) will add ``build_session_task`` etc. here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "build_augmentation_task",
    "build_holdout_task",
    "stratified_sample",
]


def stratified_sample(
    df: pd.DataFrame,
    group_col: str,
    n_per_group: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Take ``min(n_per_group, group_size)`` rows from each group.

    Why stratified instead of plain random:
        a single random sample over a heterogeneous corpus (e.g. 19 digital
        + 186 film photos in LUKI) silently under-samples the small cohort,
        and any cohort-asymmetry in the eval result then washes out into the
        mean. Stratifying forces every cohort to be represented.

    NaN groups are kept (``dropna=False``) — the digital cohort in LUKI
    has ``roll_tags == NaN`` and we definitely want it in the eval set.

    Returns a fresh DataFrame with reset index.
    """
    if n_per_group <= 0:
        raise ValueError(f"n_per_group must be positive, got {n_per_group}")
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(group_col, dropna=False):
        take = min(n_per_group, len(g))
        # Each group draws its own seed deterministically from the rng so
        # the result is reproducible even if pandas iterates groups in a
        # different order across versions.
        parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31))))
    return pd.concat(parts).reset_index(drop=True)


def build_augmentation_task(
    manifest: pd.DataFrame,
    n_per_group: int = 8,
    group_col: str = "roll_tags",
    seed: int = 0,
) -> pd.DataFrame:
    """Build the query set for the augmentation-invariance eval task.

    For this task every photo can be a query — we apply augmentation and
    expect the original to come back at top-1. We stratify on
    ``group_col`` (``roll_tags`` by default) so every cohort is represented:
    augmentation invariance can fail differently across cohorts (see the
    H2 cohort-density finding in ``03c``), and we want that visible.

    Returns
    -------
    pd.DataFrame
        Subset of ``manifest`` columns sufficient for the runner:
        ``relative_path``, ``filename``, ``file_hash``, plus the original
        manifest columns (``medium``, ``roll_tags``, etc.) so downstream
        code can group by cohort without re-joining.
    """
    required = {"relative_path", "filename", "file_hash"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    if group_col not in manifest.columns:
        raise ValueError(f"group_col {group_col!r} not in manifest columns")
    return stratified_sample(manifest, group_col, n_per_group, seed=seed)


def build_holdout_task(
    manifest: pd.DataFrame,
    group_col: str = "roll_tags",
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Build the query set for the leave-one-out hold-out eval task.

    For each photo with a non-null ``group_col`` value, we'll hold it out
    and expect the rest of its group to come back at the top. To make
    Average Precision behave as a continuous quantity (rather than a
    Bernoulli), we drop groups smaller than ``min_group_size``: with size
    < 3 the relevance set after hold-out has 0 or 1 elements and AP
    collapses.

    Photos with ``group_col == NaN`` (e.g. LUKI's digital cohort, which has
    no roll metadata) are silently *excluded* — they have no proxy
    relevance label, so this eval cannot judge them. Document the gap
    elsewhere; do not pretend they were evaluated. The augmentation task
    in ``03c`` is the cohort-blind alternative.

    Returns
    -------
    pd.DataFrame
        Subset of ``manifest`` containing only the rows whose group is
        eligible. The full manifest is still needed at eval time (for the
        relevance lookups), so the runner consumes both.
    """
    if group_col not in manifest.columns:
        raise ValueError(f"group_col {group_col!r} not in manifest columns")
    if min_group_size < 2:
        raise ValueError(
            f"min_group_size must be ≥ 2 (need at least 1 relevant doc after "
            f"hold-out to compute any retrieval metric), got {min_group_size}"
        )
    sizes = manifest.groupby(group_col, dropna=False).size()
    eligible = sizes[(sizes.index.notna()) & (sizes >= min_group_size)].index
    return manifest[manifest[group_col].isin(eligible)].reset_index(drop=True)
