"""CLI entry point: `python -m luki.app` launches the Gradio UI."""

from __future__ import annotations

import argparse

from luki.app.main import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the LUKI Gradio app.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a public temporary URL via gradio.live (72h).",
    )
    args = parser.parse_args()

    launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
