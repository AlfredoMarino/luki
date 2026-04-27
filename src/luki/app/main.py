"""Gradio UI for interactive similarity search over the LUKI photo library.

This UI is a **pure HTTP client** of the LUKI FastAPI backend. It does NOT
load the DINOv3 model or connect to Qdrant directly — all heavy lifting
happens in the API. The rationale: the model is 1.2GB in RAM, and having
two processes own it wastes 1.2GB for no benefit.

Three tabs:
    1. "Explorar biblioteca" — click a photo from the indexed dataset,
       see its top-K nearest neighbors.
    2. "Subir imagen"       — upload a new photo, embed it live,
       see the top-K nearest neighbors.
    3. "Búsqueda con filtros" — combine a query photo with metadata filters
       (medium, camera, year, session) to demo hybrid search.

All data flows through the API, which serves images as static files at
``/static/photos/...``. The Gradio Gallery widget accepts URLs directly,
so we pass the absolute URL (``{API_URL}{image_url}``) into each item.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import gradio as gr
import httpx
from PIL import Image

from luki.app import services

logger = logging.getLogger(__name__)

TOP_K_DEFAULT = 10
GALLERY_PREVIEW_LIMIT = 500  # safety cap for the browse gallery


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _format_caption(score: float, result: dict[str, Any]) -> str:
    medium = result.get("medium", "?")
    camera = result.get("camera", "?")
    session = result.get("session_name") or result.get("roll_date") or "—"
    filename = result.get("filename", "?")
    return f"{score:.3f} · {medium}/{camera} · {session} · {filename}"


def _absolute_image_url(image_url: str | None) -> str | None:
    """Turn the API-relative ``/static/photos/...`` into an absolute URL.

    Uses the **public** API URL (host-mapped) because this URL is embedded
    in HTML and resolved by the user's browser — which lives outside the
    Docker network and cannot reach internal service names like ``api``.
    """
    if not image_url:
        return None
    return f"{services.get_public_api_url()}{image_url}"


def _results_to_gallery(results: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Convert API search results into Gradio Gallery items (url, caption)."""
    items: list[tuple[str, str]] = []
    for r in results:
        url = _absolute_image_url(r.get("image_url"))
        if not url:
            continue
        items.append((url, _format_caption(r.get("score", 0.0), r)))
    return items


def _pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Serialize a PIL image to bytes for upload as multipart form-data."""
    img = image.convert("RGB") if image.mode != "RGB" else image
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# API calls (all Gradio callbacks delegate to these)                           #
# --------------------------------------------------------------------------- #


def _api_list_photos(limit: int) -> list[dict[str, Any]]:
    """GET /photos — used to populate the library gallery."""
    client = services.get_http_client()
    resp = client.get("/photos", params={"offset": 0, "limit": limit})
    resp.raise_for_status()
    return resp.json().get("photos", [])


def _api_get_filters() -> dict[str, list]:
    """GET /filters — unique values for each filterable field."""
    client = services.get_http_client()
    resp = client.get("/filters")
    resp.raise_for_status()
    return resp.json()


def _api_search_by_index(index: int, top_k: int) -> dict[str, Any]:
    """POST /search/by-index — find neighbors of a dataset photo."""
    client = services.get_http_client()
    resp = client.post(
        "/search/by-index",
        json={"index": index, "top_k": top_k},
    )
    resp.raise_for_status()
    return resp.json()


def _api_search_by_image(image_bytes: bytes, top_k: int) -> dict[str, Any]:
    """POST /search/by-image — upload an image and find neighbors."""
    client = services.get_http_client()
    resp = client.post(
        "/search/by-image",
        files={"file": ("query.jpg", image_bytes, "image/jpeg")},
        data={"top_k": str(top_k)},
    )
    resp.raise_for_status()
    return resp.json()


def _api_search_filtered(
    image_bytes: bytes,
    top_k: int,
    medium: str | None,
    camera: str | None,
    year: int | None,
    session: str | None,
) -> dict[str, Any]:
    """POST /search/filtered — hybrid search with metadata filters."""
    client = services.get_http_client()
    data: dict[str, str] = {"top_k": str(top_k)}
    if medium:
        data["medium"] = medium
    if camera:
        data["camera"] = camera
    if year:
        data["year"] = str(year)
    if session:
        data["session"] = session
    resp = client.post(
        "/search/filtered",
        files={"file": ("query.jpg", image_bytes, "image/jpeg")},
        data=data,
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Gallery sources                                                              #
# --------------------------------------------------------------------------- #


def _library_gallery_items() -> list[tuple[str, str]]:
    """All photos in the library, as gallery items. Used on startup."""
    photos = _api_list_photos(limit=GALLERY_PREVIEW_LIMIT)
    items: list[tuple[str, str]] = []
    for p in photos:
        url = _absolute_image_url(p.get("image_url"))
        if not url:
            continue
        caption = f"{p.get('medium', '?')}/{p.get('camera', '?')} · {p.get('filename', '?')}"
        items.append((url, caption))
    return items


# --------------------------------------------------------------------------- #
# Callbacks: Tab 1 — Explorar                                                  #
# --------------------------------------------------------------------------- #


def on_library_select(evt: gr.SelectData, top_k: int) -> tuple[list, str]:
    """User clicked a thumbnail in the library gallery."""
    idx = int(evt.index)
    try:
        payload = _api_search_by_index(index=idx, top_k=top_k)
    except httpx.HTTPStatusError as exc:
        return [], f"Error del API: {exc.response.status_code} — {exc.response.text}"
    except httpx.RequestError as exc:
        return [], f"No pude contactar al API: {exc}"

    gallery = _results_to_gallery(payload.get("results", []))
    header = f"**Query:** {payload.get('query', f'index={idx}')}"
    return gallery, header


# --------------------------------------------------------------------------- #
# Callbacks: Tab 2 — Upload                                                    #
# --------------------------------------------------------------------------- #


def on_upload_search(image: Image.Image | None, top_k: int) -> tuple[list, str]:
    if image is None:
        return [], "Sube una imagen para buscar."

    try:
        image_bytes = _pil_to_bytes(image)
        payload = _api_search_by_image(image_bytes, top_k=top_k)
    except httpx.HTTPStatusError as exc:
        return [], f"Error del API: {exc.response.status_code} — {exc.response.text}"
    except httpx.RequestError as exc:
        return [], f"No pude contactar al API: {exc}"

    gallery = _results_to_gallery(payload.get("results", []))
    return gallery, f"**Query:** imagen subida · {image.size[0]}×{image.size[1]} px"


# --------------------------------------------------------------------------- #
# Callbacks: Tab 3 — Filtros                                                   #
# --------------------------------------------------------------------------- #


def on_filtered_search(
    image: Image.Image | None,
    medium: str,
    camera: str,
    year: int,
    session: str,
    top_k: int,
) -> tuple[list, str]:
    if image is None:
        return [], "Sube una imagen (o elige desde la pestaña Explorar) para buscar."

    # "cualquiera" and 0 mean "no filter"
    medium_val = medium if medium and medium != "cualquiera" else None
    camera_val = camera if camera and camera != "cualquiera" else None
    year_val = int(year) if year else None
    session_val = session if session and session != "cualquiera" else None

    try:
        image_bytes = _pil_to_bytes(image)
        payload = _api_search_filtered(
            image_bytes,
            top_k=top_k,
            medium=medium_val,
            camera=camera_val,
            year=year_val,
            session=session_val,
        )
    except httpx.HTTPStatusError as exc:
        return [], f"Error del API: {exc.response.status_code} — {exc.response.text}"
    except httpx.RequestError as exc:
        return [], f"No pude contactar al API: {exc}"

    gallery = _results_to_gallery(payload.get("results", []))
    applied = [f for f in (medium_val, camera_val, year_val, session_val) if f]
    filter_desc = (
        f"medium={medium_val}, camera={camera_val}, year={year_val}, session={session_val}"
        if applied
        else "sin filtros"
    )
    return gallery, f"**Filtros aplicados:** {filter_desc} · **resultados:** {len(gallery)}"


# --------------------------------------------------------------------------- #
# Filter option helpers                                                        #
# --------------------------------------------------------------------------- #


def _filter_choices() -> dict[str, list]:
    """Fetch the filter options once at UI build time."""
    try:
        return _api_get_filters()
    except Exception as exc:
        logger.warning("Could not fetch filters from API: %s", exc)
        return {"mediums": [], "cameras": [], "years": [], "sessions": []}


# --------------------------------------------------------------------------- #
# UI construction                                                              #
# --------------------------------------------------------------------------- #


def build_app() -> gr.Blocks:
    # Fail fast if the API is not reachable — clearer than broken callbacks.
    services.check_api_health()

    filters = _filter_choices()
    medium_choices = ["cualquiera"] + filters.get("mediums", [])
    camera_choices = ["cualquiera"] + filters.get("cameras", [])
    year_choices = [0] + filters.get("years", [])  # 0 means "cualquiera"
    session_choices = ["cualquiera"] + filters.get("sessions", [])

    with gr.Blocks(title="LUKI — búsqueda visual por similitud") as app:
        gr.Markdown(
            "# 🔎 LUKI — búsqueda visual por similitud\n"
            "Fotos indexadas con **DINOv3-ViT-L** + **Qdrant** (HNSW, cosine). "
            f"UI client → API backend at `{services.get_api_url()}`."
        )

        with gr.Tabs():
            # ---------- Tab 1: Explorar ---------- #
            with gr.Tab("📚 Explorar biblioteca"):
                with gr.Row():
                    top_k_lib = gr.Slider(
                        1, 20, value=TOP_K_DEFAULT, step=1, label="Top-K vecinos"
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Biblioteca (click una foto)")
                        library = gr.Gallery(
                            value=_library_gallery_items(),
                            columns=4,
                            height=600,
                            allow_preview=True,
                            show_label=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Vecinos más cercanos")
                        lib_header = gr.Markdown("_Haz click en una foto de la izquierda._")
                        lib_results = gr.Gallery(
                            columns=3,
                            height=600,
                            allow_preview=True,
                            show_label=False,
                        )

                library.select(
                    fn=on_library_select,
                    inputs=[top_k_lib],
                    outputs=[lib_results, lib_header],
                )

            # ---------- Tab 2: Upload ---------- #
            with gr.Tab("⬆️ Subir imagen"):
                with gr.Row():
                    top_k_up = gr.Slider(
                        1, 20, value=TOP_K_DEFAULT, step=1, label="Top-K vecinos"
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        uploader = gr.Image(
                            label="Sube una foto",
                            type="pil",
                            height=500,
                        )
                        upload_btn = gr.Button("🔍 Buscar similares", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("### Vecinos más cercanos")
                        up_header = gr.Markdown("_Sube una imagen y pulsa el botón._")
                        up_results = gr.Gallery(
                            columns=3,
                            height=600,
                            allow_preview=True,
                            show_label=False,
                        )

                upload_btn.click(
                    fn=on_upload_search,
                    inputs=[uploader, top_k_up],
                    outputs=[up_results, up_header],
                )

            # ---------- Tab 3: Filtros ---------- #
            with gr.Tab("🎛️ Búsqueda con filtros"):
                with gr.Row():
                    top_k_flt = gr.Slider(
                        1, 20, value=TOP_K_DEFAULT, step=1, label="Top-K vecinos"
                    )
                with gr.Row():
                    with gr.Column(scale=1):
                        flt_uploader = gr.Image(
                            label="Sube una foto (query)",
                            type="pil",
                            height=400,
                        )
                        gr.Markdown("### Filtros híbridos")
                        medium_dd = gr.Dropdown(
                            choices=medium_choices,
                            value="cualquiera",
                            label="Medium",
                        )
                        camera_dd = gr.Dropdown(
                            choices=camera_choices,
                            value="cualquiera",
                            label="Cámara",
                        )
                        year_dd = gr.Dropdown(
                            choices=year_choices,
                            value=0,
                            label="Año (0 = cualquiera)",
                        )
                        session_dd = gr.Dropdown(
                            choices=session_choices,
                            value="cualquiera",
                            label="Sesión",
                        )
                        flt_btn = gr.Button("🔍 Buscar", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("### Resultados filtrados")
                        flt_header = gr.Markdown(
                            "_Sube una imagen, elige filtros y pulsa el botón._"
                        )
                        flt_results = gr.Gallery(
                            columns=3,
                            height=700,
                            allow_preview=True,
                            show_label=False,
                        )

                flt_btn.click(
                    fn=on_filtered_search,
                    inputs=[
                        flt_uploader,
                        medium_dd,
                        camera_dd,
                        year_dd,
                        session_dd,
                        top_k_flt,
                    ],
                    outputs=[flt_results, flt_header],
                )

    return app


def launch(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    app = build_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Soft(),
    )
