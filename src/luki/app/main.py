"""Gradio UI for interactive similarity search over the LUKI photo library.

Three tabs:
    1. "Explorar biblioteca" — click a photo from the indexed dataset,
       see its top-K nearest neighbors.
    2. "Subir imagen"       — upload a new photo, embed it live,
       see the top-K nearest neighbors.
    3. "Búsqueda con filtros" — combine a query photo with metadata filters
       (medium, camera, year, session) to demo hybrid search.

All model state lives in the `services` module as module-level singletons
(see that file for the rationale). This file is pure view logic.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
from qdrant_client.http import models as qmodels

from luki.app import services
from luki.embeddings.dataset import load_image

logger = logging.getLogger(__name__)

TOP_K_DEFAULT = 10
GALLERY_PREVIEW_LIMIT = 500  # safety cap for the browse gallery

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _format_caption(score: float, payload: dict[str, Any]) -> str:
    medium = payload.get("medium", "?")
    camera = payload.get("camera", "?")
    session = payload.get("session_name") or payload.get("roll_date") or "—"
    filename = payload.get("filename", "?")
    return f"{score:.3f} · {medium}/{camera} · {session} · {filename}"


def _results_to_gallery(results, exclude_hash: str | None = None) -> list[tuple[str, str]]:
    """Convert Qdrant search results into Gradio Gallery items."""
    items: list[tuple[str, str]] = []
    for r in results:
        p = r.payload or {}
        if exclude_hash and p.get("file_hash") == exclude_hash:
            continue
        path = p.get("absolute_path")
        if not path:
            continue
        items.append((path, _format_caption(r.score, p)))
    return items


def _build_filter(
    medium: str | None,
    camera: str | None,
    year: int | None,
    session: str | None,
) -> qmodels.Filter | None:
    """Turn UI dropdown values into a Qdrant Filter object (or None)."""
    conditions: list[qmodels.FieldCondition] = []
    if medium and medium != "cualquiera":
        conditions.append(
            qmodels.FieldCondition(key="medium", match=qmodels.MatchValue(value=medium))
        )
    if camera and camera != "cualquiera":
        conditions.append(
            qmodels.FieldCondition(key="camera", match=qmodels.MatchValue(value=camera))
        )
    if year and year != 0:
        conditions.append(
            qmodels.FieldCondition(key="year", match=qmodels.MatchValue(value=int(year)))
        )
    if session and session != "cualquiera":
        conditions.append(
            qmodels.FieldCondition(
                key="session_name", match=qmodels.MatchValue(value=session)
            )
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


# --------------------------------------------------------------------------- #
# Gallery sources                                                              #
# --------------------------------------------------------------------------- #


def _library_gallery_items() -> list[tuple[str, str]]:
    """All photos in the manifest, as gallery items. Used on startup."""
    df = services.get_manifest().head(GALLERY_PREVIEW_LIMIT)
    items: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        caption = f"{row.get('medium', '?')}/{row.get('camera', '?')} · {row.get('filename', '?')}"
        items.append((row["absolute_path"], caption))
    return items


# --------------------------------------------------------------------------- #
# Callbacks: Tab 1 — Explorar                                                  #
# --------------------------------------------------------------------------- #


def on_library_select(evt: gr.SelectData, top_k: int) -> tuple[list, str]:
    """User clicked a thumbnail in the library gallery."""
    df = services.get_manifest().head(GALLERY_PREVIEW_LIMIT)
    idx = int(evt.index)
    if idx < 0 or idx >= len(df):
        return [], "Índice fuera de rango."

    query_row = df.iloc[idx]
    try:
        img = load_image(query_row["absolute_path"])
    except Exception as exc:
        return [], f"No pude abrir la imagen: {exc}"

    embedder = services.get_embedder()
    store = services.get_store()

    query_vec = embedder.embed([img])[0]
    # +1 to absorb the self-match, which we then filter out
    results = store.search(query_vec, top_k=top_k + 1)
    gallery = _results_to_gallery(results, exclude_hash=query_row["file_hash"])[:top_k]

    header = (
        f"**Query:** `{query_row['filename']}` · "
        f"{query_row['medium']}/{query_row['camera']} · "
        f"{query_row.get('session_name') or query_row.get('roll_date') or '—'}"
    )
    return gallery, header


# --------------------------------------------------------------------------- #
# Callbacks: Tab 2 — Upload                                                    #
# --------------------------------------------------------------------------- #


def on_upload_search(image: Image.Image | None, top_k: int) -> tuple[list, str]:
    if image is None:
        return [], "Sube una imagen para buscar."

    embedder = services.get_embedder()
    store = services.get_store()

    img = image.convert("RGB") if image.mode != "RGB" else image
    query_vec = embedder.embed([img])[0]
    results = store.search(query_vec, top_k=top_k)
    gallery = _results_to_gallery(results)

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

    embedder = services.get_embedder()
    store = services.get_store()

    img = image.convert("RGB") if image.mode != "RGB" else image
    query_vec = embedder.embed([img])[0]

    query_filter = _build_filter(medium, camera, year, session)
    results = store.search(query_vec, top_k=top_k, query_filter=query_filter)
    gallery = _results_to_gallery(results)

    filter_desc = (
        f"medium={medium}, camera={camera}, year={year}, session={session}"
        if query_filter
        else "sin filtros"
    )
    return gallery, f"**Filtros aplicados:** {filter_desc} · **resultados:** {len(gallery)}"


# --------------------------------------------------------------------------- #
# Filter option helpers                                                        #
# --------------------------------------------------------------------------- #


def _distinct_values(column: str) -> list[str]:
    df = services.get_manifest()
    values = sorted({str(v) for v in df[column].dropna().unique()})
    return ["cualquiera"] + values


def _distinct_years() -> list[int]:
    df = services.get_manifest()
    years = sorted({int(v) for v in df["year"].dropna().unique()})
    return [0] + years  # 0 means "cualquiera"


# --------------------------------------------------------------------------- #
# UI construction                                                              #
# --------------------------------------------------------------------------- #


def build_app() -> gr.Blocks:
    services.warmup()

    with gr.Blocks(title="LUKI — búsqueda visual por similitud") as app:
        gr.Markdown(
            "# 🔎 LUKI — búsqueda visual por similitud\n"
            "Fotos indexadas con **DINOv3-ViT-L** + **Qdrant** (HNSW, cosine)."
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
                            choices=_distinct_values("medium"),
                            value="cualquiera",
                            label="Medium",
                        )
                        camera_dd = gr.Dropdown(
                            choices=_distinct_values("camera"),
                            value="cualquiera",
                            label="Cámara",
                        )
                        year_dd = gr.Dropdown(
                            choices=_distinct_years(),
                            value=0,
                            label="Año (0 = cualquiera)",
                        )
                        session_dd = gr.Dropdown(
                            choices=_distinct_values("session_name"),
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
