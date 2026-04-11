"""CLI entry point: `python -m luki.embeddings --config config/base.yaml`."""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from luki.embeddings.pipeline import run_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DINOv3 embeddings for all photos in the manifest "
        "and upsert them into Qdrant.",
    )
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Path to the YAML config (default: config/base.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all photos, bypassing the cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N photos (for smoke testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stats = run_embeddings(config, force=args.force, limit=args.limit)

    print()
    print("=" * 50)
    print(f"  Processed : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}  (cache hit)")
    print(f"  Failed    : {stats['failed']}")
    print(f"  Total     : {stats['total']}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
