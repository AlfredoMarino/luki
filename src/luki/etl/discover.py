"""
discover.py

Recursively discovers valid image files under a root directory.
Returns paths ready to be consumed by the ETL pipeline.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".heif"}


def discover_images(root_dir: Path, extensions: set[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """
    Recursively discovers all valid image files under root_dir.

    Skips:
        - Files with unsupported extensions
        - Hidden files and directories (starting with '.')
        - Directories that raise PermissionError

    Args:
        root_dir:   root directory to search
        extensions: set of lowercase extensions to include, e.g. {'.jpg', '.png'}

    Returns:
        Sorted list of absolute Path objects for each discovered image.

    Raises:
        FileNotFoundError: if root_dir does not exist
    """
    root_dir = Path(root_dir).resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root_dir}")

    extensions = {ext.lower() for ext in extensions}
    discovered: list[Path] = []
    skipped_permission = 0

    logger.info(f"Discovering images under: {root_dir}")
    logger.info(f"Accepted extensions: {extensions}")

    for item in _walk(root_dir):
        try:
            if item.suffix.lower() in extensions:
                discovered.append(item)
        except PermissionError:
            skipped_permission += 1
            logger.warning(f"Permission denied: {item}")

    discovered.sort()

    logger.info(
        f"Discovery complete — found {len(discovered)} images"
        + (f", skipped {skipped_permission} due to permission errors" if skipped_permission else "")
    )

    return discovered


def _walk(directory: Path):
    """
    Generator that yields all files recursively,
    skipping hidden entries and inaccessible directories.
    """
    try:
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                yield from _walk(entry)
            elif entry.is_file():
                yield entry
    except PermissionError:
        logger.warning(f"Permission denied, skipping directory: {directory}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    images = discover_images(root)

    print(f"\nFound {len(images)} images:")
    for img in images[:10]:
        print(f"  {img}")
    if len(images) > 10:
        print(f"  ... and {len(images) - 10} more")
