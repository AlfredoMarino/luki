"""Runtime services for the LUKI Gradio client.

After the API refactor, the Gradio UI is a **pure HTTP client** of the LUKI
FastAPI backend. It does NOT load the DINOv3 model or connect to Qdrant
directly — all heavy lifting happens in the API (see ``luki.api.services``).

This module provides only:
    - ``get_api_url()``      : resolves the API base URL (env-overridable).
    - ``get_http_client()``  : a long-lived ``httpx.Client`` for connection reuse.
    - ``check_api_health()`` : smoke-test the API is reachable on startup.

Environment variables:
    LUKI_API_URL         — API base URL for **server-side** httpx calls
                           (Gradio process → API process). Defaults to
                           ``http://127.0.0.1:8000``. In Docker compose set
                           to ``http://api:8000`` (internal service name).

    LUKI_PUBLIC_API_URL  — API base URL for **browser-facing** assets (image
                           ``<img src=...>`` URLs rendered in the page). The
                           browser runs on the host and cannot resolve the
                           Docker-internal ``api`` hostname, so this must be
                           the host-mapped URL (e.g. ``http://localhost:8000``).
                           Defaults to ``LUKI_API_URL`` when unset — correct
                           for bare-metal dev, wrong for Docker.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_api_url() -> str:
    """Base URL of the LUKI FastAPI backend. Override with LUKI_API_URL."""
    return os.environ.get("LUKI_API_URL", "http://127.0.0.1:8000").rstrip("/")


@lru_cache(maxsize=1)
def get_public_api_url() -> str:
    """Browser-facing API base URL (for image ``<img src>`` attributes).

    The Gradio page is served to a browser on the host machine, which lives
    outside the Docker compose network and cannot resolve container names
    like ``api``. Any URL we embed in HTML must therefore use the
    host-mapped address (e.g. ``http://localhost:8000``), not the internal
    service name used by server-side httpx calls.

    Falls back to ``LUKI_API_URL`` when not set — correct for bare-metal
    development where the UI and API share a host.
    """
    public = os.environ.get("LUKI_PUBLIC_API_URL")
    if public:
        return public.rstrip("/")
    return get_api_url()


@lru_cache(maxsize=1)
def get_http_client() -> httpx.Client:
    """Long-lived HTTP client with connection pooling.

    A persistent client avoids the TCP handshake + TLS overhead on every
    request. The 60s timeout is generous for the single-image embed path,
    which can take ~1 second on CPU.
    """
    return httpx.Client(base_url=get_api_url(), timeout=60.0)


def check_api_health(
    max_attempts: int = 60,
    delay_seconds: float = 2.0,
) -> dict:
    """Call the API's /health endpoint, retrying while it comes online.

    The API loads DINOv3 (~1.2GB weights, ~30s on CPU) before serving any
    request, so on a cold docker-compose up the UI container will beat the
    API to readiness. We poll until it responds or give up after
    ``max_attempts * delay_seconds`` seconds (default 2 min).
    """
    client = get_http_client()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get("/health")
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                "API reachable at %s — %d photos indexed, model=%s",
                get_api_url(),
                data.get("qdrant_points", 0),
                data.get("model_version", "?"),
            )
            return data
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_error = exc
            logger.info(
                "Waiting for API at %s (attempt %d/%d): %s",
                get_api_url(),
                attempt,
                max_attempts,
                exc.__class__.__name__,
            )
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"API at {get_api_url()} did not become ready after "
        f"{max_attempts} attempts. Last error: {last_error}"
    )
