"""Pydantic request/response models for the LUKI API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class SearchByIndexRequest(BaseModel):
    """Search for similar photos using a photo from the dataset."""

    index: int = Field(..., description="Row index in the manifest (0-based)")
    top_k: int = Field(10, ge=1, le=50, description="Number of neighbors")


class FilteredSearchParams(BaseModel):
    """Optional metadata filters for hybrid search."""

    medium: str | None = Field(None, description="Filter by medium (e.g. 'digital', 'film')")
    camera: str | None = Field(None, description="Filter by camera (e.g. 'canon-500d')")
    year: int | None = Field(None, description="Filter by year (e.g. 2026)")
    session: str | None = Field(None, description="Filter by session_name")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class PhotoResult(BaseModel):
    """A single search result."""

    score: float
    filename: str | None = None
    relative_path: str | None = None
    image_url: str | None = None
    medium: str | None = None
    camera: str | None = None
    year: int | None = None
    session_name: str | None = None
    roll_date: str | None = None


class SearchResponse(BaseModel):
    """Response envelope for search endpoints."""

    query: str
    results: list[PhotoResult]
    count: int


class PhotoMeta(BaseModel):
    """Metadata for a single photo from the manifest."""

    index: int
    filename: str | None = None
    relative_path: str | None = None
    image_url: str | None = None
    medium: str | None = None
    camera: str | None = None
    year: int | None = None
    session_name: str | None = None
    roll_date: str | None = None
    film_stock: str | None = None
    film_iso: int | None = None
    width: int | None = None
    height: int | None = None


class PhotoListResponse(BaseModel):
    """Paginated list of photos."""

    photos: list[PhotoMeta]
    total: int
    offset: int
    limit: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_points: int
    manifest_photos: int
    model_version: str


class FilterOptions(BaseModel):
    """Available filter values for the UI."""

    mediums: list[str]
    cameras: list[str]
    years: list[int]
    sessions: list[str]
