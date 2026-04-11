"""Interactive demo of LUKI similarity search.

Picks a random photo from the manifest, embeds it, and queries Qdrant for the
top-K most similar photos. Also demonstrates hybrid filtered search.

Run:
    python scripts/demo_similarity_search.py
    python scripts/demo_similarity_search.py --photo-index 42
    python scripts/demo_similarity_search.py --medium film
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from qdrant_client.http import models as qmodels

from luki.embeddings.dataset import load_image, load_manifest
from luki.embeddings.model import DinoV3Embedder
from luki.embeddings.store import QdrantStore


def format_row(score: float, payload: dict) -> str:
    return (
        f"  {score:.4f}  "
        f"{payload.get('medium', '?'):<8} "
        f"{payload.get('camera', '?'):<15} "
        f"{str(payload.get('session_name') or payload.get('roll_date') or '—'):<25} "
        f"{payload.get('filename', '?')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--photo-index",
        type=int,
        default=None,
        help="Index of the query photo in the manifest (default: random)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--medium",
        choices=["digital", "film"],
        default=None,
        help="Filter results to a single medium (hybrid search demo)",
    )
    parser.add_argument(
        "--exclude-same-session",
        action="store_true",
        help="Exclude photos from the same session_name (find cross-session matches)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["data"]["processed_dir"]).resolve() / "manifest.parquet"
    df = load_manifest(manifest_path)

    # Pick query photo
    if args.photo_index is None:
        idx = int(np.random.randint(0, len(df)))
    else:
        idx = args.photo_index
    query_row = df.iloc[idx]

    print("=" * 80)
    print(f"QUERY: row {idx}/{len(df)}")
    print(f"  file     : {query_row['filename']}")
    print(f"  medium   : {query_row['medium']}")
    print(f"  camera   : {query_row['camera']}")
    print(f"  session  : {query_row.get('session_name') or query_row.get('roll_date')}")
    print(f"  path     : {query_row['absolute_path']}")
    print("=" * 80)

    # Embed the query
    embedder = DinoV3Embedder(
        model_name=config["embeddings"]["model_name"],
        device=config["embeddings"].get("device", "auto"),
    )
    img = load_image(query_row["absolute_path"])
    query_vector = embedder.embed([img])[0]  # (1024,)

    # Connect to Qdrant
    store = QdrantStore(
        url=config["qdrant"]["url"],
        collection_name=config["qdrant"]["collection_name"],
        vector_size=config["qdrant"]["vector_size"],
    )

    # --- Search 1: unfiltered top-K ---
    print(f"\n[1] Top-{args.top_k} most similar (no filter)")
    print("-" * 80)
    results = store.search(query_vector, top_k=args.top_k + 1)  # +1 to absorb self-match
    # Remove the self-match if present
    results = [r for r in results if r.payload.get("file_hash") != query_row["file_hash"]][
        : args.top_k
    ]
    for r in results:
        print(format_row(r.score, r.payload))

    # --- Search 2: filtered by medium ---
    if args.medium:
        print(f"\n[2] Top-{args.top_k} most similar restricted to medium='{args.medium}'")
        print("-" * 80)
        f = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="medium",
                    match=qmodels.MatchValue(value=args.medium),
                )
            ]
        )
        results = store.search(query_vector, top_k=args.top_k + 1, query_filter=f)
        results = [r for r in results if r.payload.get("file_hash") != query_row["file_hash"]][
            : args.top_k
        ]
        for r in results:
            print(format_row(r.score, r.payload))

    # --- Search 3: exclude same session ---
    if args.exclude_same_session and query_row.get("session_name"):
        print(f"\n[3] Top-{args.top_k} most similar EXCLUDING session='{query_row['session_name']}'")
        print("-" * 80)
        f = qmodels.Filter(
            must_not=[
                qmodels.FieldCondition(
                    key="session_name",
                    match=qmodels.MatchValue(value=query_row["session_name"]),
                )
            ]
        )
        results = store.search(query_vector, top_k=args.top_k, query_filter=f)
        for r in results:
            print(format_row(r.score, r.payload))

    print()


if __name__ == "__main__":
    main()
