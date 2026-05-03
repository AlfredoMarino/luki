"""``luki-eval`` — run the LUKI evaluation suite end-to-end.

Usage::

    luki-eval --task all --model all                      # full sweep, default params
    luki-eval --task holdout --model dinov3,clip          # one task, two models
    luki-eval --task augment --model siglip --top-k 20    # narrower scope

Output: a JSON record per (model, task) under ``data/processed/eval/``,
plus a printed headline table.

The CLI is *thin* — it orchestrates the pieces in ``eval/`` rather than
re-implementing them. If you want to change *what* the eval measures,
edit ``runner.py`` / ``queries.py`` / ``metrics.py``; this file only
wires those parts together with argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from luki.eval.augment import DEFAULT_AUGMENTATIONS
from luki.eval.embedders import (
    BaseEmbedder,
    ClipEmbedder,
    DinoEmbedder,
    SiglipEmbedder,
)
from luki.eval.metrics import bootstrap_ci
from luki.eval.queries import build_augmentation_task, build_holdout_task
from luki.eval.runner import evaluate_holdout

logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, type[BaseEmbedder]] = {
    "dinov3": DinoEmbedder,
    "clip":   ClipEmbedder,
    "siglip": SiglipEmbedder,
}
ALL_MODELS = list(MODEL_REGISTRY.keys())
ALL_TASKS = ["holdout", "augment"]


# --------------------------------------------------------------------------- #
# Embedding pipeline                                                          #
# --------------------------------------------------------------------------- #


def _embed_corpus(
    model_name: str,
    embedder: BaseEmbedder,
    manifest: pd.DataFrame,
    raw_dir: Path,
    cache_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    """Embed every photo in the manifest. Cached to ``cache_dir/{model}.npy``."""
    from luki.embeddings.dataset import load_image, resolve_path

    cache_path = cache_dir / f"{model_name}.npy"
    if use_cache and cache_path.exists():
        logger.info("[%s] loading cached embeddings: %s", model_name, cache_path)
        return np.load(cache_path)

    logger.info("[%s] embedding %d photos…", model_name, len(manifest))
    t0 = time.time()
    # Stream batches to keep peak memory bounded — film scans at 3000+ px
    # are tens of MB each; loading all 205 upfront blows the 4 GB container.
    BATCH = 16
    chunks: list[np.ndarray] = []
    paths = manifest["relative_path"].tolist()
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i : i + BATCH]
        batch_imgs = [load_image(resolve_path(p, raw_dir)) for p in batch_paths]
        chunks.append(embedder.embed(batch_imgs))
        # Free the PIL refs explicitly so the next iteration starts clean
        del batch_imgs
        if (i // BATCH) % 4 == 0:
            logger.info("[%s] %d/%d embedded", model_name, i + len(batch_paths), len(paths))
    vecs = np.concatenate(chunks, axis=0).astype(np.float32)
    logger.info("[%s] done in %.1fs, shape=%s", model_name, time.time() - t0, vecs.shape)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vecs)
    return vecs


# --------------------------------------------------------------------------- #
# Eval tasks                                                                  #
# --------------------------------------------------------------------------- #


def _run_holdout(
    embedder: BaseEmbedder,
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    *,
    group_col: str,
    min_group_size: int,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    eval_set = build_holdout_task(manifest, group_col=group_col, min_group_size=min_group_size)
    per_query = evaluate_holdout(
        eval_set=eval_set,
        embeddings=embeddings,
        corpus_paths=manifest["relative_path"].tolist(),
        corpus_manifest=manifest,
        group_col=group_col,
        top_k=top_k,
    )
    metrics = _aggregate(per_query, ["Recall@10", "Recall@20", "P@10", "RR", "AP"])
    return per_query, metrics


def _run_augmentation(
    embedder: BaseEmbedder,
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    raw_dir: Path,
    *,
    n_per_group: int,
    seed: int,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    from luki.embeddings.dataset import load_image, resolve_path

    eval_set = build_augmentation_task(manifest, n_per_group=n_per_group, seed=seed)
    corpus_paths = manifest["relative_path"].tolist()
    path_to_idx = {p: i for i, p in enumerate(corpus_paths)}

    rows: list[dict[str, Any]] = []
    for aug_name, aug_fn in DEFAULT_AUGMENTATIONS.items():
        rng = np.random.default_rng(seed)
        for _, q in eval_set.iterrows():
            original = load_image(resolve_path(q["relative_path"], raw_dir))
            aug_img = aug_fn(original, rng)
            qvec = embedder.embed([aug_img])[0]
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            sims = embeddings @ qvec
            order = np.argsort(-sims)[:top_k]
            top_paths = [corpus_paths[i] for i in order]
            target_path = corpus_paths[path_to_idx[q["relative_path"]]]
            rank = top_paths.index(target_path) + 1 if target_path in top_paths else None
            rows.append({
                "aug": aug_name,
                "filename": q["filename"],
                "medium": q["medium"],
                "roll_tags": q["roll_tags"],
                "rank": rank,
                "recall@1": 1.0 if rank == 1 else 0.0,
                "rr": 1.0 / rank if rank else 0.0,
            })
    per_query = pd.DataFrame(rows)
    metrics = _aggregate(per_query, ["recall@1", "rr"])
    # Per-augmentation breakdown
    breakdown = per_query.groupby("aug")["recall@1"].mean().to_dict()
    metrics["per_augmentation_recall@1"] = {k: float(v) for k, v in breakdown.items()}
    return per_query, metrics


def _aggregate(per_query: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    """Mean + bootstrap CI for each metric column."""
    out: dict[str, dict[str, float]] = {}
    for col in cols:
        m, lo, hi = bootstrap_ci(per_query[col].to_numpy())
        out[col] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    return out


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #


def _write_run_json(
    out_dir: Path,
    model_name: str,
    task_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    per_query: pd.DataFrame,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{model_name}_{task_name}_{timestamp}.json"
    record = {
        "model": model_name,
        "task": task_name,
        "config": config,
        "metrics": metrics,
        "per_query": per_query.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=_json_safe)
    return path


def _json_safe(obj: Any) -> Any:
    """Coerce numpy / pandas / nan into JSON-safe types."""
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is pd.NA or (isinstance(obj, float) and obj != obj):
        return None
    raise TypeError(f"unserialisable type: {type(obj).__name__}")


def _print_headline(records: list[dict[str, Any]]) -> None:
    """Compact single-table summary across all (model, task) runs."""
    print()
    print("=" * 70)
    print("LUKI eval — headline summary")
    print("=" * 70)
    by_model: dict[str, dict[str, str]] = {}
    for r in records:
        by_model.setdefault(r["model"], {})
        if r["task"] == "holdout":
            ap = r["metrics"]["AP"]
            p10 = r["metrics"]["P@10"]
            by_model[r["model"]]["holdout AP"] = f"{ap['mean']:.3f} [{ap['ci_lo']:.2f}, {ap['ci_hi']:.2f}]"
            by_model[r["model"]]["holdout P@10"] = f"{p10['mean']:.3f} [{p10['ci_lo']:.2f}, {p10['ci_hi']:.2f}]"
        elif r["task"] == "augment":
            r1 = r["metrics"]["recall@1"]
            by_model[r["model"]]["augmentation R@1"] = f"{r1['mean']:.3f} [{r1['ci_lo']:.2f}, {r1['ci_hi']:.2f}]"
    df = pd.DataFrame(by_model).T
    print(df.to_string())
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _parse_models(s: str) -> list[str]:
    if s == "all":
        return ALL_MODELS
    requested = [t.strip() for t in s.split(",")]
    bad = [t for t in requested if t not in MODEL_REGISTRY]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown model(s): {bad}; available: {ALL_MODELS}")
    return requested


def _parse_tasks(s: str) -> list[str]:
    if s == "all":
        return ALL_TASKS
    requested = [t.strip() for t in s.split(",")]
    bad = [t for t in requested if t not in ALL_TASKS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown task(s): {bad}; available: {ALL_TASKS}")
    return requested


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="luki-eval",
        description="Run the LUKI evaluation suite (holdout and/or augmentation tasks across one or more models).",
    )
    p.add_argument("--task",  default="all", help="task(s): comma-separated subset of {holdout, augment} or 'all'")
    p.add_argument("--model", default="all", help="model(s): comma-separated subset of {dinov3, clip, siglip} or 'all'")
    p.add_argument("--top-k",          type=int, default=50, help="top-k for retrieval (default 50)")
    p.add_argument("--n-per-group",    type=int, default=8,  help="per-group sample size for augmentation eval (default 8)")
    p.add_argument("--min-group-size", type=int, default=3,  help="minimum group size for holdout eval (default 3)")
    p.add_argument("--seed",           type=int, default=42, help="rng seed for sampling/augmentation (default 42)")
    p.add_argument("--group-col",      default="roll_tags",  help="metadata column used as relevance proxy (default roll_tags)")
    p.add_argument("--out", type=Path,
                   help="output dir for JSON records (default: data/processed/eval/)")
    p.add_argument("--no-cache", action="store_true",
                   help="re-embed even if a cache exists (default: use cache)")
    p.add_argument("-v", "--verbose", action="store_true", help="enable info logging")
    return p


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    models = _parse_models(args.model)
    tasks = _parse_tasks(args.task)

    # Resolve paths from the existing API services config so we share the
    # same conventions as the rest of LUKI (no duplicate config plumbing).
    # raw_dir is at <repo>/data/raw → derive repo root by walking up two levels.
    from luki.api import services
    raw_dir = services.get_raw_dir()
    repo_root = raw_dir.parent.parent
    manifest = services.get_manifest()

    out_dir = args.out or (repo_root / "data" / "processed" / "eval")
    cache_dir = repo_root / "data" / "processed" / "eval_embeddings"

    common_config = {
        "top_k": args.top_k,
        "n_per_group": args.n_per_group,
        "min_group_size": args.min_group_size,
        "group_col": args.group_col,
        "seed": args.seed,
        "n_corpus": len(manifest),
    }

    records: list[dict[str, Any]] = []
    for model_name in models:
        logger.info("=== model: %s ===", model_name)
        embedder_cls = MODEL_REGISTRY[model_name]
        embedder = embedder_cls()
        embeddings = _embed_corpus(
            model_name=model_name, embedder=embedder, manifest=manifest,
            raw_dir=raw_dir, cache_dir=cache_dir, use_cache=not args.no_cache,
        )

        for task in tasks:
            logger.info("--- task: %s ---", task)
            if task == "holdout":
                per_query, metrics = _run_holdout(
                    embedder, embeddings, manifest,
                    group_col=args.group_col,
                    min_group_size=args.min_group_size,
                    top_k=args.top_k,
                )
            elif task == "augment":
                per_query, metrics = _run_augmentation(
                    embedder, embeddings, manifest, raw_dir,
                    n_per_group=args.n_per_group,
                    seed=args.seed,
                    top_k=args.top_k,
                )
            else:
                raise ValueError(f"unreachable: unknown task {task!r}")

            run_path = _write_run_json(
                out_dir=out_dir, model_name=model_name, task_name=task,
                config=common_config, metrics=metrics, per_query=per_query,
            )
            print(f"  wrote {run_path.relative_to(repo_root) if run_path.is_relative_to(repo_root) else run_path}")
            records.append({"model": model_name, "task": task, "metrics": metrics})

    _print_headline(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
