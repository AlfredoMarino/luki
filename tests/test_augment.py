"""Unit tests for ``luki.eval.augment`` and ``luki.eval.queries``.

The strategy: every test pins a property a future change is likely to
break. Reproducibility under seed is the single most important contract,
so byte-equality tests come first. Stratified sampling tests come second.
Cohort diagnostics get smoke tests against synthetic vectors with known
structure.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from luki.eval.augment import (
    DEFAULT_AUGMENTATIONS,
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
from luki.eval.queries import build_augmentation_task, stratified_sample


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_image() -> Image.Image:
    """A 100×80 RGB image with deterministic pixel content.

    Using a gradient (rather than random pixels) keeps the test
    deterministic without needing a separate seed for the fixture itself.
    """
    arr = np.zeros((80, 100, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, 100, dtype=np.uint8)[None, :]  # red gradient
    arr[..., 1] = np.linspace(0, 255, 80, dtype=np.uint8)[:, None]   # green gradient
    arr[..., 2] = 128
    return Image.fromarray(arr, "RGB")


def _img_bytes(img: Image.Image) -> bytes:
    """Serialize an image to bytes for byte-equality comparison.

    Using a deterministic encoder (PNG, no compression metadata) means two
    visually-identical images produce identical bytes.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Augmentation reproducibility — the headline contract                        #
# --------------------------------------------------------------------------- #


class TestAugmentationDeterminism:
    """Same input + same seed must produce byte-identical output, every time."""

    @pytest.mark.parametrize("name", list(DEFAULT_AUGMENTATIONS.keys()))
    def test_same_seed_same_output(self, sample_image: Image.Image, name: str):
        fn = DEFAULT_AUGMENTATIONS[name]
        a = fn(sample_image, np.random.default_rng(42))
        b = fn(sample_image, np.random.default_rng(42))
        assert _img_bytes(a) == _img_bytes(b), f"{name} not deterministic"

    @pytest.mark.parametrize("name", ["crop", "brightness", "rotate"])
    def test_different_seed_different_output(self, sample_image: Image.Image, name: str):
        # grayscale is excluded — it has no randomness, output is the same
        # under any seed, by design.
        fn = DEFAULT_AUGMENTATIONS[name]
        a = fn(sample_image, np.random.default_rng(1))
        b = fn(sample_image, np.random.default_rng(2))
        assert _img_bytes(a) != _img_bytes(b), (
            f"{name} produced identical output for different seeds — "
            f"the rng isn't actually being consumed"
        )

    def test_grayscale_is_seed_invariant(self, sample_image: Image.Image):
        # grayscale takes rng for signature uniformity but ignores it.
        a = grayscale(sample_image, np.random.default_rng(1))
        b = grayscale(sample_image, np.random.default_rng(999))
        assert _img_bytes(a) == _img_bytes(b)


# --------------------------------------------------------------------------- #
# Augmentation output shape & semantics                                       #
# --------------------------------------------------------------------------- #


class TestAugmentationOutputs:
    def test_grayscale_returns_3channel_rgb(self, sample_image: Image.Image):
        out = grayscale(sample_image, np.random.default_rng(0))
        assert out.mode == "RGB"
        assert out.size == sample_image.size
        # All three channels equal → grayscale-in-RGB
        arr = np.array(out)
        assert np.array_equal(arr[..., 0], arr[..., 1])
        assert np.array_equal(arr[..., 1], arr[..., 2])

    def test_random_crop_dimensions(self, sample_image: Image.Image):
        out = random_crop(sample_image, np.random.default_rng(0), ratio=0.7)
        w, h = sample_image.size
        # Allow ±1 px slack for int-rounding
        assert abs(out.size[0] - int(w * 0.7)) <= 1
        assert abs(out.size[1] - int(h * 0.7)) <= 1

    def test_random_crop_rejects_invalid_ratio(self, sample_image: Image.Image):
        with pytest.raises(ValueError):
            random_crop(sample_image, np.random.default_rng(0), ratio=0.0)
        with pytest.raises(ValueError):
            random_crop(sample_image, np.random.default_rng(0), ratio=1.5)

    def test_brightness_jitter_preserves_shape(self, sample_image: Image.Image):
        out = brightness_jitter(sample_image, np.random.default_rng(0))
        assert out.size == sample_image.size
        assert out.mode == sample_image.mode

    def test_rotate_preserves_shape(self, sample_image: Image.Image):
        # expand=False means the canvas size stays the same.
        out = rotate(sample_image, np.random.default_rng(0))
        assert out.size == sample_image.size

    def test_rotate_with_zero_angle_is_near_identity(self, sample_image: Image.Image):
        # angle=0 should round-trip the image (modulo PIL's resampling).
        out = rotate(sample_image, np.random.default_rng(0), angle_range=(0.0, 0.0))
        # Pixel-perfect round-trip isn't guaranteed (rotation by 0 is still
        # done by the rotation kernel), so we settle for "very close".
        diff = np.abs(np.array(out, dtype=int) - np.array(sample_image, dtype=int))
        assert diff.mean() < 1.0  # mean per-pixel intensity diff < 1


# --------------------------------------------------------------------------- #
# stratified_sample / build_augmentation_task                                 #
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    """A 12-row manifest spanning 4 cohorts of size {3, 3, 4, 2}."""
    return pd.DataFrame({
        "relative_path": [f"p{i}.jpg" for i in range(12)],
        "filename":      [f"p{i}.jpg" for i in range(12)],
        "file_hash":     [f"h{i}" for i in range(12)],
        "medium":        ["film"] * 10 + ["digital"] * 2,
        "roll_tags":     ["a"] * 3 + ["b"] * 3 + ["c"] * 4 + [None] * 2,
    })


class TestStratifiedSample:
    def test_caps_per_group(self, toy_manifest: pd.DataFrame):
        # n=2 per group → groups of size {2, 2, 2, 2} = 8 rows.
        out = stratified_sample(toy_manifest, "roll_tags", n_per_group=2, seed=0)
        assert len(out) == 8
        assert (out.groupby("roll_tags", dropna=False).size() == 2).all()

    def test_takes_all_when_group_smaller_than_n(self, toy_manifest: pd.DataFrame):
        # n=10 per group, but groups are size {3, 3, 4, 2} → returns all 12 rows.
        out = stratified_sample(toy_manifest, "roll_tags", n_per_group=10, seed=0)
        assert len(out) == 12

    def test_keeps_nan_groups(self, toy_manifest: pd.DataFrame):
        # NaN cohort (digital) must NOT be silently dropped.
        out = stratified_sample(toy_manifest, "roll_tags", n_per_group=2, seed=0)
        assert out["roll_tags"].isna().sum() == 2

    def test_reproducible_under_seed(self, toy_manifest: pd.DataFrame):
        a = stratified_sample(toy_manifest, "roll_tags", n_per_group=2, seed=42)
        b = stratified_sample(toy_manifest, "roll_tags", n_per_group=2, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_picks_different_rows(self, toy_manifest: pd.DataFrame):
        # When n < group_size for at least one group, different seeds should
        # pick different members of that group.
        a = stratified_sample(toy_manifest, "roll_tags", n_per_group=1, seed=1)
        b = stratified_sample(toy_manifest, "roll_tags", n_per_group=1, seed=2)
        # There's a small chance two seeds happen to pick the same rows —
        # if this test ever flakes, change the seeds.
        assert not a.equals(b)

    def test_rejects_zero_n(self, toy_manifest: pd.DataFrame):
        with pytest.raises(ValueError):
            stratified_sample(toy_manifest, "roll_tags", n_per_group=0)


class TestBuildAugmentationTask:
    def test_returns_query_set(self, toy_manifest: pd.DataFrame):
        out = build_augmentation_task(toy_manifest, n_per_group=2, seed=0)
        # Must carry the columns the runner needs
        for col in ("relative_path", "filename", "file_hash"):
            assert col in out.columns

    def test_validates_required_columns(self):
        bad = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_augmentation_task(bad)

    def test_validates_group_col(self, toy_manifest: pd.DataFrame):
        with pytest.raises(ValueError, match="group_col"):
            build_augmentation_task(toy_manifest, group_col="nope")


# --------------------------------------------------------------------------- #
# Diagnostics                                                                 #
# --------------------------------------------------------------------------- #


class TestIntraCohortSimilarity:
    def test_dense_cohort_scores_higher(self):
        # Build two cohorts: A is tightly clustered, B is spread out.
        rng = np.random.default_rng(0)
        center_a = rng.standard_normal(8)
        center_a /= np.linalg.norm(center_a)
        # A: 5 vectors near center_a (small noise)
        a_vecs = center_a + 0.05 * rng.standard_normal((5, 8))
        # B: 5 random unit vectors (spread out)
        b_vecs = rng.standard_normal((5, 8))
        # Normalize both to unit norm
        a_vecs /= np.linalg.norm(a_vecs, axis=1, keepdims=True)
        b_vecs /= np.linalg.norm(b_vecs, axis=1, keepdims=True)
        vecs = np.vstack([a_vecs, b_vecs])
        cohorts = ["A"] * 5 + ["B"] * 5

        out = intra_cohort_similarity(vecs, cohorts)
        a_sim = out[out["cohort"] == "A"]["mean_intra_cos"].iloc[0]
        b_sim = out[out["cohort"] == "B"]["mean_intra_cos"].iloc[0]
        assert a_sim > b_sim, f"dense cohort A should score higher: {a_sim} vs {b_sim}"
        assert a_sim > 0.9  # near-collinear by construction

    def test_singleton_cohort_returns_nan(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = intra_cohort_similarity(vecs, ["A", "B"])
        assert out["mean_intra_cos"].isna().all()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            intra_cohort_similarity(np.eye(3), ["A", "B"])


class TestNearestNeighborSimilarity:
    def test_basic_shape(self):
        vecs = np.eye(4)
        out = nearest_neighbor_similarity(vecs, ["A", "A", "B", "B"])
        assert len(out) == 4
        assert "nn_sim" in out.columns
        # Orthogonal unit vectors → all 1-NN sims are 0
        assert (out["nn_sim"] == 0).all()

    def test_near_duplicate_pair_shows_high_sim(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(8); v /= np.linalg.norm(v)
        # Two near-copies + 3 random vectors
        vecs = np.vstack([
            v,
            v + 0.01 * rng.standard_normal(8),
            rng.standard_normal(8),
            rng.standard_normal(8),
            rng.standard_normal(8),
        ])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        out = nearest_neighbor_similarity(vecs, ["pair", "pair", "rand", "rand", "rand"])
        # The pair members should have 1-NN sim near 1
        pair = out[out["cohort"] == "pair"]["nn_sim"].to_numpy()
        rand = out[out["cohort"] == "rand"]["nn_sim"].to_numpy()
        assert pair.min() > 0.99
        assert rand.max() < pair.min()


class TestFailureForensics:
    def test_annotates_query_and_winner_cohorts(self):
        manifest = pd.DataFrame({
            "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
            "filename":      ["a.jpg", "b.jpg", "c.jpg"],
            "roll_tags":     ["roll1", "roll1", None],
            "medium":        ["film", "film", "digital"],
        })
        results = pd.DataFrame({
            "filename":   ["a.jpg", "c.jpg"],
            "roll_tags":  ["roll1", None],
            "medium":     ["film", "digital"],
            "top1_path":  ["b.jpg", "a.jpg"],
        })
        out = annotate_failure_forensics(results, manifest)
        # query a (roll1) → winner b (roll1) → same cohort
        assert out.iloc[0]["query_cohort"] == "roll1"
        assert out.iloc[0]["winner_cohort"] == "roll1"
        assert out.iloc[0]["same_cohort"] is True or out.iloc[0]["same_cohort"] == True
        # query c (digital, NaN roll) → winner a (roll1) → different cohort
        assert out.iloc[1]["query_cohort"] == "digital"
        assert out.iloc[1]["winner_cohort"] == "roll1"
        assert not out.iloc[1]["same_cohort"]

    def test_requires_top1_path(self):
        with pytest.raises(ValueError, match="top1_path"):
            annotate_failure_forensics(pd.DataFrame({"foo": [1]}), pd.DataFrame())
