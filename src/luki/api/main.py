"""FastAPI application for LUKI visual similarity search.

Endpoints provide REST access to the functionality: search by dataset index,
upload an image, or combine with metadata filters.

All model state is managed by the shared singletons in ``luki.api.services``.
The API is the **sole owner** of the DINOv3 model — the Gradio UI is an HTTP
client of this service, not a co-tenant of the model.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from qdrant_client.http import models as qmodels

from luki.api.schemas import (
    FilteredSearchParams,
    FilterOptions,
    HealthResponse,
    PhotoListResponse,
    PhotoMeta,
    PhotoResult,
    SearchByIndexRequest,
    SearchResponse,
)
from luki.api import services
from luki.embeddings.dataset import load_image, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: warm up singletons once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eager initialization of all singletons on startup."""
    services.warmup()
    yield


app = FastAPI(
    title="LUKI API",
    description="Visual similarity search over a personal photo library. "
    "Powered by DINOv3-ViT-L embeddings and Qdrant HNSW.",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount raw photos as static files so the API can serve images.
# Done at module level (not in lifespan) because FastAPI does NOT execute
# on_event("startup") handlers when a lifespan context manager is also
# defined — they are mutually exclusive APIs. Mounting here runs during
# app construction, which is fine because the raw_dir path is resolved
# from config and does not require the model to be loaded.
app.mount(
    "/static/photos",
    StaticFiles(directory=str(services.get_raw_dir())),
    name="photos",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_url(relative_path: str | None) -> str | None:
    """Build a URL for a photo from its relative_path."""
    if not relative_path:
        return None
    # relative_path already uses forward slashes (POSIX)
    return f"/static/photos/{relative_path}"


def _payload_to_result(score: float, payload: dict[str, Any]) -> PhotoResult:
    """Convert a Qdrant payload dict into a PhotoResult schema."""
    year = payload.get("year")
    return PhotoResult(
        score=score,
        filename=payload.get("filename"),
        relative_path=payload.get("relative_path"),
        image_url=_image_url(payload.get("relative_path")),
        medium=payload.get("medium"),
        camera=payload.get("camera"),
        year=int(year) if year is not None else None,
        session_name=payload.get("session_name"),
        roll_date=payload.get("roll_date"),
    )


def _build_filter(params: FilteredSearchParams) -> qmodels.Filter | None:
    """Build a Qdrant filter from the request parameters."""
    conditions: list[qmodels.FieldCondition] = []
    if params.medium:
        conditions.append(
            qmodels.FieldCondition(
                key="medium", match=qmodels.MatchValue(value=params.medium)
            )
        )
    if params.camera:
        conditions.append(
            qmodels.FieldCondition(
                key="camera", match=qmodels.MatchValue(value=params.camera)
            )
        )
    if params.year:
        conditions.append(
            qmodels.FieldCondition(
                key="year", match=qmodels.MatchValue(value=params.year)
            )
        )
    if params.session:
        conditions.append(
            qmodels.FieldCondition(
                key="session_name", match=qmodels.MatchValue(value=params.session)
            )
        )
    return qmodels.Filter(must=conditions) if conditions else None


def _row_to_meta(idx: int, row) -> PhotoMeta:
    """Convert a manifest DataFrame row to a PhotoMeta schema."""
    import pandas as pd

    def _clean(val):
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            return val.item()
        return val

    return PhotoMeta(
        index=idx,
        filename=_clean(row.get("filename")),
        relative_path=_clean(row.get("relative_path")),
        image_url=_image_url(_clean(row.get("relative_path"))),
        medium=_clean(row.get("medium")),
        camera=_clean(row.get("camera")),
        year=int(row["year"]) if not pd.isna(row.get("year")) else None,
        session_name=_clean(row.get("session_name")),
        roll_date=_clean(row.get("roll_date")),
        film_stock=_clean(row.get("film_stock")),
        film_iso=int(row["film_iso"]) if not pd.isna(row.get("film_iso")) else None,
        width=int(row["width"]) if not pd.isna(row.get("width")) else None,
        height=int(row["height"]) if not pd.isna(row.get("height")) else None,
    )


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Service health check: Qdrant count, manifest size, model version."""
    store = services.get_store()
    manifest = services.get_manifest()
    embedder = services.get_embedder()
    return HealthResponse(
        status="ok",
        qdrant_points=store.count(),
        manifest_photos=len(manifest),
        model_version=embedder.model_version,
    )


@app.get("/photos", response_model=PhotoListResponse)
async def list_photos(
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=1000, description="Page size"),
):
    """Paginated list of photos from the manifest."""
    df = services.get_manifest()
    total = len(df)
    page = df.iloc[offset : offset + limit]
    photos = [_row_to_meta(offset + i, row) for i, (_, row) in enumerate(page.iterrows())]
    return PhotoListResponse(photos=photos, total=total, offset=offset, limit=limit)


@app.get("/photos/{index}", response_model=PhotoMeta)
async def get_photo(index: int):
    """Metadata for a single photo by manifest index."""
    df = services.get_manifest()
    if index < 0 or index >= len(df):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range (0-{len(df)-1})")
    row = df.iloc[index]
    return _row_to_meta(index, row)


@app.get("/filters", response_model=FilterOptions)
async def get_filters():
    """Available values for each filterable field."""
    df = services.get_manifest()
    return FilterOptions(
        mediums=sorted({str(v) for v in df["medium"].dropna().unique()}),
        cameras=sorted({str(v) for v in df["camera"].dropna().unique()}),
        years=sorted({int(v) for v in df["year"].dropna().unique()}),
        sessions=sorted({str(v) for v in df["session_name"].dropna().unique()}),
    )


# ---------------------------------------------------------------------------
# POST endpoints — search
# ---------------------------------------------------------------------------


@app.post("/search/by-index", response_model=SearchResponse)
async def search_by_index(req: SearchByIndexRequest):
    """Find similar photos using a photo already in the dataset."""
    df = services.get_manifest()
    if req.index < 0 or req.index >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Index {req.index} out of range (0-{len(df)-1})",
        )

    query_row = df.iloc[req.index]
    raw_dir = services.get_raw_dir()
    photo_path = resolve_path(query_row["relative_path"], raw_dir)

    try:
        img = load_image(photo_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot open image: {exc}")

    embedder = services.get_embedder()
    store = services.get_store()

    query_vec = embedder.embed([img])[0]
    results = store.search(query_vec, top_k=req.top_k + 1)

    # Exclude self-match
    file_hash = query_row["file_hash"]
    photo_results = [
        _payload_to_result(r.score, r.payload or {})
        for r in results
        if (r.payload or {}).get("file_hash") != file_hash
    ][: req.top_k]

    return SearchResponse(
        query=f"index={req.index} ({query_row['filename']})",
        results=photo_results,
        count=len(photo_results),
    )


@app.post("/search/by-image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = Form(10, ge=1, le=50),
):
    """Upload an image and find the most similar photos in the library."""
    try:
        img = Image.open(file.file).convert("RGB")
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")

    embedder = services.get_embedder()
    store = services.get_store()

    query_vec = embedder.embed([img])[0]
    results = store.search(query_vec, top_k=top_k)

    photo_results = [
        _payload_to_result(r.score, r.payload or {}) for r in results
    ]

    return SearchResponse(
        query=f"uploaded image ({img.size[0]}x{img.size[1]}px)",
        results=photo_results,
        count=len(photo_results),
    )


@app.post("/search/filtered", response_model=SearchResponse)
async def search_filtered(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = Form(10, ge=1, le=50),
    medium: str | None = Form(None),
    camera: str | None = Form(None),
    year: int | None = Form(None),
    session: str | None = Form(None),
):
    """Upload an image and search with metadata filters (hybrid search)."""
    try:
        img = Image.open(file.file).convert("RGB")
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")

    embedder = services.get_embedder()
    store = services.get_store()

    query_vec = embedder.embed([img])[0]
    params = FilteredSearchParams(
        medium=medium, camera=camera, year=year, session=session
    )
    query_filter = _build_filter(params)
    results = store.search(query_vec, top_k=top_k, query_filter=query_filter)

    photo_results = [
        _payload_to_result(r.score, r.payload or {}) for r in results
    ]

    filter_desc = (
        f"medium={medium}, camera={camera}, year={year}, session={session}"
        if query_filter
        else "none"
    )

    return SearchResponse(
        query=f"uploaded image ({img.size[0]}x{img.size[1]}px) | filters: {filter_desc}",
        results=photo_results,
        count=len(photo_results),
    )
