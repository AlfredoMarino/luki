# LUKI — Lightweight Unsupervised Knowledge from Images

Visual similarity search over a personal photo library, powered by **DINOv3-ViT-L** embeddings and **Qdrant** vector database.

## Architecture

```
photos/            ETL              DINOv3-ViT-L         Qdrant (HNSW)
  digital/ ──────► manifest   ──────► 1024-dim   ──────► cosine search
  film/            .parquet          embeddings           + payload filters
                                                              │
                                                    ┌─────────┴──────────┐
                                                    ▼                    ▼
                                              FastAPI :8000        Gradio :7860
                                              (REST API)           (interactive UI)
```

**Pipeline steps:**
1. **ETL** — Discover photos, parse folder conventions (`medium/year/camera/session`), extract EXIF metadata, persist `manifest.parquet`.
2. **Embeddings** — Load each photo, forward pass through DINOv3-ViT-L (303M params), L2-normalize to 1024-dim vector, upsert into Qdrant with metadata payload.
3. **Search** — Given a query image (or dataset index), embed it and find top-K nearest neighbors via HNSW cosine similarity. Optionally filter by medium, camera, year, or session.

## Requirements

- Python 3.10+
- Docker (for Qdrant, and optionally for the full stack)
- HuggingFace account with access to DINOv3 (the model is gated):
  1. Accept the license at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
  2. Create a read token at https://huggingface.co/settings/tokens
  3. **Local use**: run `huggingface-cli login` once to cache the token
  4. **Docker use**: copy `.env.example` to `.env` and paste your token as `HF_TOKEN`

## Quick start (local)

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Start Qdrant
docker compose up qdrant -d

# 3. Run pipelines
luki-etl                # discover photos → manifest.parquet
luki-embed              # embed photos → Qdrant (first run: ~2 min on CPU)

# 4. Launch interfaces
luki-app                # Gradio UI  → http://localhost:7860
luki-api                # FastAPI    → http://localhost:8000
                        # Swagger UI → http://localhost:8000/docs
```

## Quick start (Docker)

```bash
# 1. Provide your HuggingFace token (needed to download DINOv3)
cp .env.example .env
# then edit .env and paste HF_TOKEN=hf_...

# 2. Start the full stack
docker compose up -d
# API     → http://localhost:8000
# Gradio  → http://localhost:7860
# Qdrant  → http://localhost:6333/dashboard
```

> **Memory note:** DINOv3 uses ~1.2GB RAM. Running `api` + `app` simultaneously
> requires ~2.4GB just for the model. For demos, consider running only one:
> ```bash
> docker compose up qdrant api -d    # API only
> ```

## CLI reference

| Command | Description | Key flags |
|---------|-------------|-----------|
| `luki-etl` | Run ETL pipeline | `--config PATH` |
| `luki-embed` | Generate embeddings + upsert to Qdrant | `--config PATH`, `--force`, `--limit N` |
| `luki-app` | Launch Gradio UI | `--host`, `--port`, `--share` |
| `luki-api` | Launch FastAPI server | `--host`, `--port`, `--reload` |
| `luki-eval` | Run evaluation suite (see Evaluation section below) | `--task`, `--model`, `--top-k`, `--seed`, `--out` |

All commands also work as `python -m luki.{etl.cli,embeddings,app,api}`.

## API reference

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Qdrant count, manifest size, model version |
| `GET` | `/photos?offset=0&limit=20` | Paginated photo list from manifest |
| `GET` | `/photos/{index}` | Single photo metadata |
| `GET` | `/filters` | Unique values for medium, camera, year, session |
| `POST` | `/search/by-index` | Search by dataset index (JSON: `{index, top_k}`) |
| `POST` | `/search/by-image` | Upload image → find similar (multipart form) |
| `POST` | `/search/filtered` | Upload image + metadata filters (multipart form) |

## Evaluation

LUKI ships with a quantitative evaluation of its embedding quality so the choice of DINOv3 over CLIP / SigLIP is *defensible with numbers*, not aesthetic preference. Two complementary tasks measure two different properties of the embedding space:

| Task | What it tests | Relevance signal |
|---|---|---|
| **Augmentation invariance** | Does the model embed augmented copies of the same photo near the original? | The original is the only relevant doc per query (no proxy bias). |
| **Roll hold-out** | Does the model cluster photos a human would call "from the same trip"? | Same `roll_tags` value (a weak metadata proxy). |

Both run via brute-force cosine over an in-memory embedding matrix (the corpus is ~200 photos, ~1 MB at 1024 dims) — no Qdrant round-trip per query, and the same runner serves all three models without re-indexing.

### Headline numbers

Run on the 205-photo reference corpus. CIs are 95% bootstraps over per-query results (10k resamples). See `notebooks/03e_model_comparison.ipynb` for the full analysis.

| Model | dim | Holdout AP | Holdout P@10 | Augmentation R@1 |
|---|---|---|---|---|
| DINOv3-ViT-L | 1024 | 0.180 [0.17, 0.19] | 0.421 [0.39, 0.45] | **0.990** [0.97, 1.00] |
| **CLIP-ViT-B/32** | 512 | **0.219** [0.20, 0.23] | **0.495** [0.47, 0.52] | 0.969 [0.94, 0.99] |
| SigLIP-base/16 | 768 | **0.219** [0.20, 0.24] | 0.476 [0.45, 0.51] | **0.990** [0.97, 1.00] |

**Reading the table.** DINOv3 wins augmentation invariance — that's its training objective, so this is a sanity check, not a verdict. CLIP and SigLIP win the roll-task: their image–text contrastive training induces *semantic* grouping that happens to align with how rolls are organized (one trip = one roll = related scenes). For LUKI's product story ("find the rest of my trip"), CLIP is the better default — but DINOv3 stays competitive and is more robust to perturbation. Bootstrap CIs overlap between CLIP and SigLIP on both tasks; the difference between them is not statistically distinguishable on this corpus.

> Reproduce: `luki-eval --task all --model all` writes a JSON per (model, task) under `data/processed/eval/` and prints the summary table. First run downloads CLIP (~600 MB) and SigLIP (~875 MB) into the HF cache; subsequent runs use cached corpus embeddings (~1 MB each).

### Limitations (named honestly)

- **Roll labels are weak.** Same-roll photos are correlated by *time* (one developer date) and *camera*, which usually but not always implies visual similarity. False positives (unrelated subjects in the same roll) and false negatives (similar subjects across rolls) both depress AP. We report alongside Recall@k so you can read which way the noise pulls.
- **The 19 digital photos have no roll.** They are silently excluded from the holdout task — *not* averaged in as zeros, which would understate the model. The augmentation task evaluates them.
- **Bootstrap CIs assume i.i.d. queries.** Within-roll queries are correlated; the cluster bootstrap (resample whole rolls) would be more honest. We use the simpler resample-photos variant; CIs are mildly optimistic.
- **Sample size.** ~186 holdout queries × 5 rolls. Enough to claim trends, not enough to claim small differences (overlapping CIs ≠ statistically distinguishable).

The notebook series `03a–03e` walks through every choice above with a senior-track teaching narrative.

## Project structure

```
luki/
├── config/
│   └── base.yaml                 # All configuration (paths, model, Qdrant)
├── data/
│   ├── raw/                      # Photo library (digital/ + film/)
│   ├── processed/                # manifest.parquet + summary
│   └── qdrant/                   # Qdrant storage (Docker volume)
├── src/luki/
│   ├── etl/
│   │   ├── discover.py           # Recursive image discovery
│   │   ├── path_parser.py        # Folder convention parser
│   │   ├── extract.py            # EXIF + file metadata extraction
│   │   ├── pipeline.py           # ETL orchestrator
│   │   └── cli.py                # CLI entry point
│   ├── embeddings/
│   │   ├── model.py              # DinoV3Embedder (torch.inference_mode)
│   │   ├── store.py              # QdrantStore wrapper (HNSW, payload indexes)
│   │   ├── dataset.py            # Manifest loader + batch iterator
│   │   ├── pipeline.py           # Embeddings orchestrator (resumable)
│   │   └── __main__.py           # CLI entry point
│   ├── app/
│   │   ├── services.py           # Shared singletons (model, store, config)
│   │   ├── main.py               # Gradio UI (3 tabs)
│   │   └── __main__.py           # CLI entry point
│   └── api/
│       ├── schemas.py            # Pydantic request/response models
│       ├── main.py               # FastAPI app + endpoints
│       └── __main__.py           # CLI entry point
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Configuration

All settings live in `config/base.yaml`:

```yaml
data:
  raw_dir: "data/raw"             # Photo library root
  processed_dir: "data/processed" # ETL output
embeddings:
  model_name: "facebook/dinov3-vitl16-pretrain-lvd1689m"
  batch_size: 8
  device: "auto"                  # "auto" → CUDA if available, else CPU
  normalize: true                 # L2-normalize embeddings
qdrant:
  url: "http://localhost:6334"    # gRPC endpoint
  collection_name: "luki_photos"
  vector_size: 1024
  distance: "Cosine"
```

Docker networking: set `LUKI_QDRANT_URL=http://qdrant:6334` to override `qdrant.url` (done automatically by docker-compose).

## Tech stack

| Component | Technology |
|-----------|-----------|
| Embeddings | DINOv3-ViT-L (facebook/dinov3-vitl16-pretrain-lvd1689m) |
| Vector DB | Qdrant v1.12 (HNSW, cosine, payload indexes) |
| REST API | FastAPI + uvicorn |
| Interactive UI | Gradio |
| ML framework | PyTorch + HuggingFace Transformers |
| Data | pandas + pyarrow (parquet) |
