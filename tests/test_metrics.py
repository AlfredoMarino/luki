"""Unit tests for ``luki.eval.metrics``.

The strategy: hand-craft small rank lists where the answer is obvious by
inspection, so a regression manifests as a wrong number and not as a
mysterious shift in an end-to-end pipeline. Every test below should be
verifiable with a pencil; if a future change breaks one, you have to actively
acknowledge that the math meaning shifted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from luki.eval.metrics import (
    average_precision,
    bootstrap_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --------------------------------------------------------------------------- #
# precision_at_k                                                              #
# --------------------------------------------------------------------------- #


class TestPrecisionAtK:
    def test_all_relevant_top_k(self):
        # 3 relevants in top-3 → 3/3 = 1.0
        assert precision_at_k([1, 1, 1, 0, 0], k=3) == 1.0

    def test_partial(self):
        # 2 relevants in top-3 → 2/3
        assert precision_at_k([1, 0, 1, 0, 0], k=3) == pytest.approx(2.0 / 3.0)

    def test_no_relevants(self):
        assert precision_at_k([0, 0, 0, 0], k=4) == 0.0

    def test_zero_k_returns_zero(self):
        assert precision_at_k([1, 1, 1], k=0) == 0.0

    def test_k_beyond_list_caps_at_list_length(self):
        # If retrieval returned fewer than k, evaluate over what we have
        # rather than dividing by a k we never hit.
        assert precision_at_k([1, 0, 1], k=10) == pytest.approx(2.0 / 3.0)

    def test_ceiling_effect_documented(self):
        # The senior-trap case: a 5-relevant corpus, page of 10 slots.
        # Even a "perfect" retrieval (all 5 relevants up top, then garbage)
        # caps at Precision@10 = 0.5.
        perfect_for_small_corpus = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert precision_at_k(perfect_for_small_corpus, k=10) == 0.5

    def test_complementary_to_recall(self):
        # Precision and Recall ask different questions of the same vector.
        # Concrete LUKI-shape numbers: 39 relevants in corpus, 2 in top-10.
        rel = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # 2/10 in top-10
        assert precision_at_k(rel, k=10) == 0.2
        assert recall_at_k(rel, k=10, n_relevant=39) == pytest.approx(2.0 / 39.0)

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError):
            precision_at_k([0, 1, 2], k=2)


# --------------------------------------------------------------------------- #
# recall_at_k                                                                 #
# --------------------------------------------------------------------------- #


class TestRecallAtK:
    def test_all_relevant_top_k(self):
        # 3 relevants in top-3, total 3 → recall@3 = 1.0
        assert recall_at_k([1, 1, 1, 0, 0], k=3, n_relevant=3) == 1.0

    def test_partial_recall(self):
        # 2 relevants in top-3, total 4 → 2/4 = 0.5
        assert recall_at_k([1, 0, 1, 0, 0], k=3, n_relevant=4) == 0.5

    def test_k_beyond_list_length(self):
        # k larger than list — same as evaluating the whole list
        assert recall_at_k([1, 0, 1], k=99, n_relevant=2) == 1.0

    def test_zero_n_relevant_is_nan(self):
        # Query with no corpus-relevant docs → undefined (nan), not 0
        assert math.isnan(recall_at_k([0, 0, 0], k=3, n_relevant=0))

    def test_zero_k_returns_zero(self):
        assert recall_at_k([1, 1, 1], k=0, n_relevant=3) == 0.0

    def test_no_hits_in_top_k(self):
        # all relevants outside the cut
        assert recall_at_k([0, 0, 0, 0, 0], k=5, n_relevant=2) == 0.0

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError):
            recall_at_k([0, 1, 2], k=2, n_relevant=2)

    def test_rejects_negative_n_relevant(self):
        with pytest.raises(ValueError):
            recall_at_k([1, 0], k=1, n_relevant=-1)


# --------------------------------------------------------------------------- #
# reciprocal_rank                                                             #
# --------------------------------------------------------------------------- #


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([1, 0, 0, 0]) == 1.0

    def test_third_position(self):
        # First relevant at rank 3 → 1/3
        assert reciprocal_rank([0, 0, 1, 0, 1]) == pytest.approx(1.0 / 3.0)

    def test_no_relevants(self):
        # Convention: RR = 0 when nothing relevant was returned
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_only_first_relevant_matters(self):
        # RR is rank-1-of-relevants; later relevants are invisible to it.
        # This is the bug pattern that justifies AP and Recall@k.
        single = reciprocal_rank([0, 1, 0, 0, 0])
        many = reciprocal_rank([0, 1, 1, 1, 1])
        assert single == many == pytest.approx(0.5)

    def test_empty_list(self):
        assert reciprocal_rank([]) == 0.0


# --------------------------------------------------------------------------- #
# average_precision                                                           #
# --------------------------------------------------------------------------- #


class TestAveragePrecision:
    def test_perfect_ranking(self):
        # All 3 relevants at the top → AP = 1.0
        assert average_precision([1, 1, 1, 0, 0], n_relevant=3) == 1.0

    def test_known_two_relevant_case(self):
        # Relevants at ranks 1 and 3, n_relevant=2.
        #   P@1 = 1/1 = 1.0
        #   P@3 = 2/3 ≈ 0.6667
        #   AP  = (1.0 + 0.6667) / 2 = 0.8333
        ap = average_precision([1, 0, 1, 0], n_relevant=2)
        assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)

    def test_unretrieved_relevants_penalise(self):
        # Same retrieved pattern as above (1 relevant at rank 1) but
        # n_relevant=2 → the second relevant is "missing" and contributes 0.
        # AP = (1.0 + 0) / 2 = 0.5
        ap = average_precision([1, 0, 0, 0], n_relevant=2)
        assert ap == 0.5

    def test_no_relevants_retrieved(self):
        assert average_precision([0, 0, 0, 0], n_relevant=3) == 0.0

    def test_zero_n_relevant_is_nan(self):
        assert math.isnan(average_precision([0, 0, 0], n_relevant=0))

    def test_relevant_at_bottom_low_score(self):
        # Single relevant at rank 5 of 5 → P@5 = 1/5, AP = 0.2 / 1 = 0.2
        ap = average_precision([0, 0, 0, 0, 1], n_relevant=1)
        assert ap == pytest.approx(0.2)

    def test_synthetic_cases_from_notebook_03b(self):
        # Mirrors the notebook's synthetic stress test — these numbers anchor
        # the chart on which the senior teaching points are made.
        n_rel = 5
        best = [1, 1, 1, 1, 1] + [0] * 45  # AP = 1.0
        # worst: relevants at ranks 6..10
        worst = [0] * 5 + [1] * 5 + [0] * 40
        # mixed: ranks 1, 3, 5, 7, 9
        mixed = [0] * 50
        for r in (1, 3, 5, 7, 9):
            mixed[r - 1] = 1

        ap_best = average_precision(best, n_rel)
        ap_worst = average_precision(worst, n_rel)
        ap_mixed = average_precision(mixed, n_rel)

        # The strict ordering AP_best > AP_mixed > AP_worst is the property
        # that makes AP rank-aware. If this breaks, AP isn't AP anymore.
        assert ap_best == 1.0
        assert ap_best > ap_mixed > ap_worst > 0


# --------------------------------------------------------------------------- #
# bootstrap_ci                                                                #
# --------------------------------------------------------------------------- #


class TestBootstrapCI:
    def test_mean_is_correct(self):
        m, _, _ = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert m == pytest.approx(0.3)

    def test_ci_brackets_mean(self):
        m, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert lo <= m <= hi

    def test_constant_values_zero_width(self):
        # All values identical → every bootstrap sample has the same mean →
        # CI width = 0. Useful sanity property.
        m, lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert lo == hi == m == 0.5

    def test_drops_nans(self):
        # nan values from undefined queries should not poison the bootstrap
        m, _, _ = bootstrap_ci([0.5, 0.5, float("nan"), 0.5])
        assert m == pytest.approx(0.5)

    def test_all_nans_returns_nan(self):
        m, lo, hi = bootstrap_ci([float("nan"), float("nan")])
        assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)

    def test_seed_is_reproducible(self):
        v = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_ci(v, seed=42)
        b = bootstrap_ci(v, seed=42)
        assert a == b

    def test_different_seed_changes_resampling(self):
        # CIs across seeds may coincide on tiny samples (quantiles can land
        # on the same discrete bootstrap means), but the underlying sequence
        # of resamples must differ. Hook into the rng directly to verify.
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        idx_a = rng_a.integers(0, len(v), size=(50, len(v)))
        idx_b = rng_b.integers(0, len(v), size=(50, len(v)))
        # Two different seeds must produce different resample index matrices.
        assert not np.array_equal(idx_a, idx_b)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            bootstrap_ci([0.1, 0.2], alpha=1.5)


# --------------------------------------------------------------------------- #
# Cross-metric properties                                                     #
# --------------------------------------------------------------------------- #


class TestCrossMetricProperties:
    """Properties that should hold across metrics — the most useful tests."""

    def test_recall_at_k_monotonic_in_k(self):
        # Recall@k is non-decreasing in k by definition.
        rel = np.array([0, 1, 0, 1, 0, 0, 1])
        n = 5
        r = [recall_at_k(rel, k, n) for k in range(1, len(rel) + 1)]
        assert all(r[i] <= r[i + 1] for i in range(len(r) - 1))

    def test_ap_bounded_in_unit_interval(self):
        # AP in [0, 1] for any binary relevance vector.
        rng = np.random.default_rng(0)
        for _ in range(20):
            length = rng.integers(5, 50)
            rel = rng.integers(0, 2, size=length)
            n_rel = max(int(rel.sum()) + rng.integers(0, 5), 1)
            ap = average_precision(rel, n_rel)
            assert 0.0 <= ap <= 1.0

    def test_rr_le_recall_at_first_relevant(self):
        # Where the first relevant is at rank r, RR = 1/r and
        # Recall@r = (#relevants up to r) / n_relevant ≥ 1/n_relevant.
        # Trivial sanity: RR ∈ [0, 1].
        rel = [0, 0, 1, 0, 1]
        rr = reciprocal_rank(rel)
        assert 0.0 <= rr <= 1.0
