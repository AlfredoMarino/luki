"""
extract.py

Extracts file metadata and EXIF data from image files.
Never raises on missing or malformed EXIF — missing fields are returned as None.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational

logger = logging.getLogger(__name__)

# EXIF tag names we care about → output field name
_EXIF_FIELD_MAP = {
    "DateTimeOriginal": "datetime_original",
    "Make": "camera_make",
    "Model": "camera_model",
    "FocalLength": "focal_length_mm",
    "FNumber": "aperture",
    "ISOSpeedRatings": "iso",
    "ExposureTime": "exposure_time",
}

# GPS sub-IFD tags
_GPS_LATITUDE = 2
_GPS_LATITUDE_REF = 1
_GPS_LONGITUDE = 4
_GPS_LONGITUDE_REF = 3


def extract_metadata(image_path: Path) -> Optional[dict[str, Any]]:
    """
    Extracts all available metadata from an image file.

    Combines:
        - File-level info  (path, size, hash)
        - Image properties (width, height, format, mode)
        - EXIF data        (camera, settings, datetime, GPS)

    Args:
        image_path: absolute path to the image

    Returns:
        dict with all metadata fields, or None if the file cannot be opened.
        Missing EXIF fields are included as None — never omitted.
    """
    image_path = Path(image_path)

    try:
        file_meta = _extract_file_meta(image_path)
        image_meta = _extract_image_meta(image_path)
        exif_meta = _extract_exif(image_path)
    except Exception as exc:
        logger.warning(f"Failed to extract metadata from {image_path}: {exc}")
        return None

    return {**file_meta, **image_meta, **exif_meta}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_file_meta(path: Path) -> dict[str, Any]:
    """File-level metadata — no Pillow needed."""
    stat = path.stat()
    return {
        "absolute_path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "file_hash": _md5(path),
    }


def _extract_image_meta(path: Path) -> dict[str, Any]:
    """Image dimensions and format via Pillow."""
    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "color_mode": img.mode,
        }


def _extract_exif(path: Path) -> dict[str, Any]:
    """
    Extracts EXIF fields. Returns a flat dict with all expected fields.
    Missing or unreadable values are set to None — never raises.
    """
    result: dict[str, Any] = {field: None for field in _EXIF_FIELD_MAP.values()}
    result["gps_lat"] = None
    result["gps_lon"] = None

    try:
        with Image.open(path) as img:
            raw_exif = img._getexif()  # returns None if no EXIF
    except Exception:
        return result  # no EXIF at all — return all-None dict

    if not raw_exif:
        return result

    # Build a human-readable exif dict: tag_name → value
    exif: dict[str, Any] = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, str(tag_id))
        exif[tag_name] = value

    # Map known fields
    for exif_key, output_key in _EXIF_FIELD_MAP.items():
        raw_value = exif.get(exif_key)
        result[output_key] = _clean_exif_value(output_key, raw_value)

    # GPS
    gps_info = exif.get("GPSInfo")
    if gps_info:
        result["gps_lat"], result["gps_lon"] = _parse_gps(gps_info)

    return result


def _clean_exif_value(field_name: str, value: Any) -> Any:
    """
    Converts raw EXIF values to clean Python types.

    EXIF stores rationals as tuples (numerator, denominator).
    This converts them to float where appropriate.
    """
    if value is None:
        return None

    # ISOSpeedRatings can come as a tuple in some cameras
    if field_name == "iso" and isinstance(value, (list, tuple)):
        value = value[0]

    # IFDRational: Pillow's rational type — convert to float early
    if isinstance(value, IFDRational):
        value = (value.numerator, value.denominator)

    # Rational values (stored as tuples by Pillow)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if den == 0:
            return None
        result = num / den
        if field_name == "focal_length_mm":
            return round(result, 1)
        if field_name == "aperture":
            return round(result, 1)
        if field_name == "exposure_time":
            # Store as fraction string for readability: '1/250'
            return f"{num}/{den}" if num == 1 else round(result, 6)
        return result

    # Strings: strip null bytes and whitespace
    if isinstance(value, str):
        return value.strip().rstrip("\x00") or None

    # Bytes: decode if possible
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").strip().rstrip("\x00") or None
        except UnicodeDecodeError:
            return None

    return value


def _parse_gps(gps_info: dict) -> tuple[Optional[float], Optional[float]]:
    """Converts raw GPSInfo IFD to decimal degrees (lat, lon)."""
    try:
        lat = _dms_to_decimal(gps_info[_GPS_LATITUDE], gps_info[_GPS_LATITUDE_REF])
        lon = _dms_to_decimal(gps_info[_GPS_LONGITUDE], gps_info[_GPS_LONGITUDE_REF])
        return lat, lon
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """
    Converts degrees/minutes/seconds tuple to decimal degrees.

    Args:
        dms: ((deg_n, deg_d), (min_n, min_d), (sec_n, sec_d))
        ref: 'N', 'S', 'E', or 'W'
    """
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60
    seconds = dms[2][0] / dms[2][1] / 3600
    decimal = degrees + minutes + seconds
    if ref in ("S", "W"):
        decimal = -decimal
    return round(decimal, 6)


def _md5(path: Path, chunk_size: int = 8192) -> str:
    """Computes MD5 hash of a file in chunks (memory-safe for large files)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python extract.py <image_path>")
        sys.exit(1)

    meta = extract_metadata(Path(sys.argv[1]))
    print(json.dumps(meta, indent=2, default=str))
