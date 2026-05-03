"""Unit tests for ``luki.eval.runner.evaluate_holdout`` and the
related ``build_holdout_task`` query builder.

Strategy: build tiny synthetic embedding matrices whose cluster structure
is known, so the metrics they produce are verifiable by hand. Two corpora:

    * **perfect_clusters** — each group's photos share a unit vector
      identical up to tiny noise; cosine similarity is ~1.0 within group,
      ~0 across. Every query should retrieve ALL its group-mates first.
      Expected: Recall@k = 1.0 once k ≥ n_relevant, AP = 1.0.

    * **scrambled_clusters** — embeddings are random, no cluster structure.
      Expected: AP near random baseline, no claim of perfection.

These two anchor the runner's behaviour: if the matrix says "perfect
groups," the runner had better return 1.0; if the matrix says "noise,"
the runner had better not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from luki.eval.queries import build_holdout_task
from luki.eval.runner import evaluate_holdout


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_corpus():
    """A 12-photo corpus with 3 groups of 4, perfectly clustered embeddings.

    Each group's photos share a unit-norm centre vector with tiny gaussian
    noise. The runner should be able to retrieve all 3 group-mates first
    for every query.
    """
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((3, 16))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vecs = []
    for c in centres:
        for _ in range(4):
            v = c + 0.001 * rng.standard_normal(16)  # very tight cluster
            vecs.append(v / np.linalg.norm(v))
    embeddings = np.array(vecs)

    paths = [f"g{g}_p{i}.jpg" for g in range(3) for i in range(4)]
    manifest = pd.DataFrame({
        "relative_path": paths,
        "filename":      paths,
        "file_hash":     [f"h_{p}" for p in paths],
        "group":         [f"group_{g}" for g in range(3) for _ in range(4)],
    })
    return embeddings, paths, manifest


@pytest.fixture
def scrambled_corpus():
    """Same shape as synthetic_corpus, but with no cluster structure."""
    rng = np.random.default_rng(123)
    vecs = rng.standard_normal((12, 16))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    paths = [f"g{g}_p{i}.jpg" for g in range(3) for i in range(4)]
    manifest = pd.DataFrame({
        "relative_path": paths,
        "filename":      paths,
        "file_hash":     [f"h_{p}" for p in paths],
        "group":         [f"group_{g}" for g in range(3) for _ in range(4)],
    })
    return vecs, paths, manifest


# --------------------------------------------------------------------------- #
# evaluate_holdout — perfect-cluster anchor                                   #
# --------------------------------------------------------------------------- #


class TestPerfectClusters:
    def test_recall_at_3_is_one(self, synthetic_corpus):
        # Every query has 3 group-mates after self-exclusion. With perfect
        # clusters, all 3 land in top-3 → Recall@k = 1.0 for k ≥ 3.
        embeddings, paths, manifest = synthetic_corpus
        results = evaluate_holdout(
            eval_set=manifest, embeddings=embeddings, corpus_paths=paths,
            corpus_manifest=manifest, group_col="group", top_k=10,
        )
        # n_relevant should be 3 for every row (4 group-mates - 1 = 3 after self-exclude)
        assert (results["n_relevant"] == 3).all()
        # Recall@10 must be 1.0 — all relevants in the top
        assert (results["Recall@10"] == 1.0).all()

    def test_ap_is_one(self, synthetic_corpus):
        # Perfect clustering → all 3 relevants at top-3 → P@k=1.0 at each
        # rank where a relevant lands → AP = 1.0.
        embeddings, paths, manifest = synthetic_corpus
        results = evaluate_holdout(
            eval_set=manifest, embeddings=embeddings, corpus_paths=paths,
            corpus_manifest=manifest, group_col="group", top_k=10,
        )
        assert (results["AP"] == 1.0).all()

    def test_rr_is_one(self, synthetic_corpus):
        embeddings, paths, manifest = synthetic_corpus
        results = evaluate_holdout(
            eval_set=manifest, embeddings=embeddings, corpus_paths=paths,
            corpus_manifest=manifest, group_col="group", top_k=10,
        )
        assert (results["RR"] == 1.0).all()

    def test_top1_is_same_group(self, synthetic_corpus):
        # Forensic check: top-1 should always be a group-mate
        embeddings, paths, manifest = synthetic_corpus
        results = evaluate_holdout(
            eval_set=manifest, embeddings=embeddings, corpus_paths=paths,
            corpus_manifest=manifest, group_col="group", top_k=10,
        )
        path_to_group = dict(zip(manifest["relative_path"], manifest["group"]))
        for _, r in results.iterrows():
            top1_group = path_to_group.get(r["top1_path"])
            assert top1_group == r["group"]


# --------------------------------------------------------------------------- #
# evaluate_holdout — scrambled (no clusters) anchor                           #
# --------------------------------------------------------------------------- #


class TestScrambledClusters:
    def test_no_perfect_recall(self, scrambled_corpus):
        # Random embeddings → can't always pull all 3 group-mates into top-10
        # (well, top-10 contains 11 docs minus self = 11; but probabilistically
        # the model that hasn't been trained will not reliably retrieve all 3)
        embeddings, paths, manifest = scrambled_corpus
        results = evaluate_holdout(
            eval_set=manifest, embeddings=embeddings, corpus_paths=paths,
            corpus_manifest=manifest, group_col="group", top_k=10,
        )
        # AP should not be perfect on average — random ordering of relevants
        assert results["AP"].mean() < 1.0


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


class TestRunnerValidation:
    def test_rejects_mismatched_embeddings_paths(self, synthetic_corpus):
        embeddings, paths, manifest = synthetic_corpus
        with pytest.raises(ValueError, match="must match"):
            evaluate_holdout(
                eval_set=manifest, embeddings=embeddings, corpus_paths=paths[:-1],
                corpus_manifest=manifest, group_col="group",
            )

    def test_rejects_missing_columns(self, synthetic_corpus):
        embeddings, paths, manifest = synthetic_corpus
        # Strip 'file_hash' column
        bad = manifest.drop(columns=["file_hash"])
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_holdout(
                eval_set=bad, embeddings=embeddings, corpus_paths=paths,
                corpus_manifest=manifest, group_col="group",
            )

    def test_rejects_query_outside_corpus(self, synthetic_corpus):
        embeddings, paths, manifest = synthetic_corpus
        # Eval set has a query whose relative_path isn't in corpus_paths
        bad = manifest.copy()
        bad.loc[0, "relative_path"] = "not_in_corpus.jpg"
        with pytest.raises(ValueError, match="not found in corpus_paths"):
            evaluate_holdout(
                eval_set=bad, embeddings=embeddings, corpus_paths=paths,
                corpus_manifest=manifest, group_col="group",
            )

    def test_rejects_non_2d_embeddings(self, synthetic_corpus):
        _, paths, manifest = synthetic_corpus
        with pytest.raises(ValueError, match="2-D"):
            evaluate_holdout(
                eval_set=manifest, embeddings=np.zeros(12), corpus_paths=paths,
                corpus_manifest=manifest, group_col="group",
            )


# --------------------------------------------------------------------------- #
# build_holdout_task                                                          #
# --------------------------------------------------------------------------- #


class TestBuildHoldoutTask:
    @pytest.fixture
    def manifest_with_mixed_groups(self) -> pd.DataFrame:
        return pd.DataFrame({
            "relative_path": [f"p{i}" for i in range(10)],
            "filename":      [f"p{i}" for i in range(10)],
            "file_hash":     [f"h{i}" for i in range(10)],
            # Sizes: A=4, B=3, C=2, D=1, NaN=0 (only A and B pass min_size=3)
            "roll_tags":     ["A"]*4 + ["B"]*3 + ["C"]*2 + [None]*1,
        })

    def test_drops_small_groups(self, manifest_with_mixed_groups):
        out = build_holdout_task(manifest_with_mixed_groups, min_group_size=3)
        groups_kept = set(out["roll_tags"].unique())
        assert groups_kept == {"A", "B"}
        assert len(out) == 7  # 4 + 3

    def test_drops_nan_groups(self, manifest_with_mixed_groups):
        out = build_holdout_task(manifest_with_mixed_groups, min_group_size=2)
        # NaN cohort never has a relevance label, must be dropped regardless
        # of whether its size would qualify.
        assert out["roll_tags"].isna().sum() == 0

    def test_min_size_threshold_below_2_rejected(self, manifest_with_mixed_groups):
        with pytest.raises(ValueError, match="must be"):
            build_holdout_task(manifest_with_mixed_groups, min_group_size=1)

    def test_unknown_group_col_rejected(self, manifest_with_mixed_groups):
        with pytest.raises(ValueError, match="not in manifest"):
            build_holdout_task(manifest_with_mixed_groups, group_col="nope")
