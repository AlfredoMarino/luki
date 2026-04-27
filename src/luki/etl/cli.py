"""CLI entry point: ``luki-etl`` / ``python -m luki.etl.cli``.

Runs the full ETL pipeline (discover -> parse path -> extract metadata ->
persist manifest).
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from luki.etl.pipeline import run_etl
from luki.utils.paths import config_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LUKI ETL pipeline: discover photos, extract "
        "metadata, and persist manifest.parquet.",
    )
    parser.add_argument(
        "--config",
        default=str(config_path()),
        help="Path to the YAML config (default: config/base.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = run_etl(config)

    if not df.empty:
        print(f"\nManifest shape: {df.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
