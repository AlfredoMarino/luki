"""
path_parser.py

Extracts structured metadata from LUKI's folder convention.

Convention:
    digital/{year}/{camera}/{session_folder}/{filename}
    film/{year}/{camera}/{YYYYMMDD}_{stock}_{iso}[_{tag-tag}]/{filename}

Separators:
    _  →  separates distinct fields
    -  →  separates words within the same field (e.g. film stock, tags)

Special values in roll folders:
    stock "x" → None (unknown film stock)
    ISO   0   → None (unknown film ISO)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_MEDIUMS = {"digital", "film"}


@dataclass
class PhotoPath:
    """Structured metadata extracted from the folder path."""

    medium: str
    year: int
    camera: str
    absolute_path: Path
    roll_date: Optional[str] = None        # film only — YYYYMMDD
    film_stock: Optional[str] = None       # film only — e.g. 'ilford-hp5'
    film_iso: Optional[int] = None         # film only — e.g. 400
    roll_tags: list[str] = field(default_factory=list)  # film only, may be empty
    session_name: Optional[str] = None   # digital only — e.g. '20260201_chile-performers'


def parse_photo_path(path: Path, root: Path) -> PhotoPath:
    """
    Extracts structured metadata from a photo's path
    according to the LUKI folder convention.

    Args:
        path: absolute path to the photo file
        root: root path of the collection (data/raw)

    Returns:
        PhotoPath with all parsed fields

    Raises:
        ValueError: if the path does not follow the expected convention
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError(f"Path {path} is not under root {root}")

    parts = relative.parts  # e.g. ('film', '2025', 'nikon_f50', '20250515_kodak_400', 're001.jpg')

    if len(parts) < 3:
        raise ValueError(
            f"Path too short to follow LUKI convention: {relative}"
        )

    medium = parts[0].lower()
    if medium not in VALID_MEDIUMS:
        raise ValueError(
            f"Unknown medium '{medium}'. Expected one of {VALID_MEDIUMS}"
        )

    try:
        year = int(parts[1])
    except ValueError:
        raise ValueError(f"Year folder '{parts[1]}' is not a valid integer")

    camera = parts[2].lower()

    if medium == "digital":
        # digital/{year}/{camera}/{session_folder}/{filename}
        if len(parts) != 5:
            raise ValueError(
                f"Digital path should have exactly 5 parts, got {len(parts)}: {relative}"
            )
        return PhotoPath(
            medium=medium,
            year=year,
            camera=camera,
            absolute_path=path,
            session_name=parts[3],
        )

    # film/{year}/{camera}/{roll_folder}/{filename}
    if len(parts) != 5:
        raise ValueError(
            f"Film path should have exactly 5 parts, got {len(parts)}: {relative}"
        )

    roll_folder = parts[3]
    roll_data = _parse_roll_folder(roll_folder)

    return PhotoPath(
        medium=medium,
        year=year,
        camera=camera,
        absolute_path=path,
        roll_date=roll_data["roll_date"],
        film_stock=roll_data["film_stock"],
        film_iso=roll_data["film_iso"],
        roll_tags=roll_data["roll_tags"],
    )


def _parse_roll_folder(folder_name: str) -> dict:
    """
    Parses a film roll folder name.

    Format: {YYYYMMDD}_{stock}_{iso}[_{tag-tag}[_{tag-tag}...]]

    Rules:
        - parts[0] → roll_date   (8 digits, YYYYMMDD)
        - parts[1] → film_stock  (may contain hyphens, e.g. 'ilford-hp5')
        - parts[2] → film_iso    (integer)
        - parts[3+] → roll_tags  (each part split by '-' and flattened)

    Args:
        folder_name: roll folder name, e.g. '20250515_kodak_400_pink-madrid'

    Returns:
        dict with keys: roll_date, film_stock, film_iso, roll_tags

    Raises:
        ValueError: if the folder name does not match the expected format
    """
    parts = folder_name.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Roll folder '{folder_name}' must have at least 3 fields: "
            f"date, stock, iso. Got {len(parts)}."
        )

    # --- date ---
    roll_date = parts[0]
    if not roll_date.isdigit() or len(roll_date) != 8:
        raise ValueError(
            f"Roll date '{roll_date}' must be 8 digits in YYYYMMDD format"
        )

    # --- stock ---
    film_stock = parts[1]  # hyphens are allowed: 'ilford-hp5', 'kodak', 'fujifilm'
    if not film_stock:
        raise ValueError(f"Film stock is empty in folder '{folder_name}'")
    if film_stock == "x":
        film_stock = None

    # --- iso ---
    try:
        film_iso = int(parts[2])
    except ValueError:
        raise ValueError(
            f"ISO value '{parts[2]}' in folder '{folder_name}' is not a valid integer"
        )
    if film_iso == 0:
        film_iso = None

    # --- tags (optional) ---
    # Each remaining part is a hyphen-separated group of words
    # e.g. ['pink-madrid', 'street-rain'] → ['pink', 'madrid', 'street', 'rain']
    roll_tags: list[str] = []
    for tag_group in parts[3:]:
        roll_tags.extend(tag_group.split("-"))

    return {
        "roll_date": roll_date,
        "film_stock": film_stock,
        "film_iso": film_iso,
        "roll_tags": roll_tags,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== _parse_roll_folder ===")
    cases = [
        "20250415_fujifilm_100",
        "20250515_kodak_400_pink-madrid",
        "20250601_ilford-hp5_400",
        "20250601_ilford-hp5_400_street-barcelona-rain",
        "20251202_x_0_paloma-salsa",
    ]
    for case in cases:
        result = _parse_roll_folder(case)
        print(f"{case}\n  → {result}\n")

    print("=== parse_photo_path ===")
    root = Path("/data/raw")
    paths = [
        root / "digital/2026/canon_500d/20260201_chile-performers/photo_001.jpg",
        root / "film/2025/canon_prima_ii/20250415_fujifilm_100/re001.jpg",
        root / "film/2025/nikon_f50/20250515_kodak_400_pink-madrid/re001.jpg",
        root / "film/2025/nikon_f50/20251202_x_0_paloma-salsa/re001.jpg",
    ]
    for p in paths:
        result = parse_photo_path(p, root)
        print(f"{p.relative_to(root)}\n  → {result}\n")
