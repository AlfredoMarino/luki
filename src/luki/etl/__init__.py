"""
luki.etl

ETL pipeline for the LUKI photo collection.

Modules:
    discover     — finds image files recursively
    path_parser  — extracts metadata from folder structure
    extract      — extracts EXIF and file metadata
    pipeline     — orchestrates the full ETL run
"""

from luki.etl.pipeline import run_etl
from luki.etl.discover import discover_images
from luki.etl.extract import extract_metadata
from luki.etl.path_parser import parse_photo_path, PhotoPath

__all__ = ["run_etl", "discover_images", "extract_metadata", "parse_photo_path", "PhotoPath"]
