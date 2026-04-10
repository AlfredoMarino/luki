"""
tests/test_path_parser.py

Unit tests for luki.etl.path_parser.
These tests cover the parsing logic without touching the filesystem.
"""

import pytest
from pathlib import Path

from luki.etl.path_parser import parse_photo_path, _parse_roll_folder, PhotoPath

ROOT = Path("/data/raw")


# ============================================================
# _parse_roll_folder
# ============================================================

class TestParseRollFolder:

    def test_basic_no_tags(self):
        result = _parse_roll_folder("20250415_fujifilm_100")
        assert result["roll_date"] == "20250415"
        assert result["film_stock"] == "fujifilm"
        assert result["film_iso"] == 100
        assert result["roll_tags"] == []

    def test_stock_with_hyphen(self):
        result = _parse_roll_folder("20250601_ilford-hp5_400")
        assert result["film_stock"] == "ilford-hp5"
        assert result["film_iso"] == 400
        assert result["roll_tags"] == []

    def test_single_tag_group(self):
        result = _parse_roll_folder("20250515_kodak_400_pink-madrid")
        assert result["roll_date"] == "20250515"
        assert result["film_stock"] == "kodak"
        assert result["film_iso"] == 400
        assert result["roll_tags"] == ["pink", "madrid"]

    def test_multiple_tag_groups(self):
        result = _parse_roll_folder("20250601_ilford-hp5_400_street-barcelona-rain")
        assert result["roll_tags"] == ["street", "barcelona", "rain"]

    def test_single_word_tag(self):
        result = _parse_roll_folder("20250601_kodak_400_portraits")
        assert result["roll_tags"] == ["portraits"]

    def test_invalid_date_not_digits(self):
        with pytest.raises(ValueError, match="8 digits"):
            _parse_roll_folder("2025-04-15_fujifilm_100")

    def test_invalid_date_wrong_length(self):
        with pytest.raises(ValueError, match="8 digits"):
            _parse_roll_folder("202504_fujifilm_100")

    def test_invalid_iso_not_integer(self):
        with pytest.raises(ValueError, match="not a valid integer"):
            _parse_roll_folder("20250415_kodak_fast")

    def test_unknown_stock_maps_to_none(self):
        result = _parse_roll_folder("20251202_x_400")
        assert result["film_stock"] is None
        assert result["film_iso"] == 400

    def test_unknown_iso_maps_to_none(self):
        result = _parse_roll_folder("20251202_kodak_0")
        assert result["film_stock"] == "kodak"
        assert result["film_iso"] is None

    def test_both_unknown(self):
        result = _parse_roll_folder("20251202_x_0_paloma-salsa")
        assert result["film_stock"] is None
        assert result["film_iso"] is None
        assert result["roll_tags"] == ["paloma", "salsa"]

    def test_missing_fields(self):
        with pytest.raises(ValueError, match="at least 3 fields"):
            _parse_roll_folder("20250415_kodak")


# ============================================================
# parse_photo_path — digital
# ============================================================

class TestParsePhotoPathDigital:

    def test_digital_basic(self):
        path = ROOT / "digital/2026/canon_500d/20260201_session/photo_001.jpg"
        result = parse_photo_path(path, ROOT)

        assert isinstance(result, PhotoPath)
        assert result.medium == "digital"
        assert result.year == 2026
        assert result.camera == "canon_500d"
        assert result.session_name == "20260201_session"
        assert result.roll_date is None
        assert result.film_stock is None
        assert result.film_iso is None
        assert result.roll_tags == []
        assert result.absolute_path == path

    def test_digital_session_name_preserved(self):
        path = ROOT / "digital/2026/canon-500d/20260201_chile-performers/3L6A9776.jpg"
        result = parse_photo_path(path, ROOT)
        assert result.session_name == "20260201_chile-performers"
        assert result.camera == "canon-500d"

    def test_digital_invalid_year(self):
        path = ROOT / "digital/twentysix/canon_500d/20260201_session/photo_001.jpg"
        with pytest.raises(ValueError, match="not a valid integer"):
            parse_photo_path(path, ROOT)

    def test_digital_too_many_parts(self):
        path = ROOT / "digital/2026/canon_500d/session/subfolder/photo_001.jpg"
        with pytest.raises(ValueError, match="exactly 5 parts"):
            parse_photo_path(path, ROOT)

    def test_unknown_medium(self):
        path = ROOT / "analog/2026/canon_500d/20260201_session/photo_001.jpg"
        with pytest.raises(ValueError, match="Unknown medium"):
            parse_photo_path(path, ROOT)

    def test_path_not_under_root(self):
        path = Path("/other/location/photo.jpg")
        with pytest.raises(ValueError, match="not under root"):
            parse_photo_path(path, ROOT)


# ============================================================
# parse_photo_path — film
# ============================================================

class TestParsePhotoPathFilm:

    def test_film_basic(self):
        path = ROOT / "film/2025/canon_prima_ii/20250415_fujifilm_100/re001.jpg"
        result = parse_photo_path(path, ROOT)

        assert result.medium == "film"
        assert result.year == 2025
        assert result.camera == "canon_prima_ii"
        assert result.roll_date == "20250415"
        assert result.film_stock == "fujifilm"
        assert result.film_iso == 100
        assert result.roll_tags == []
        assert result.session_name is None

    def test_film_with_tags(self):
        path = ROOT / "film/2025/nikon_f50/20250515_kodak_400_pink-madrid/re001.jpg"
        result = parse_photo_path(path, ROOT)

        assert result.film_stock == "kodak"
        assert result.film_iso == 400
        assert result.roll_tags == ["pink", "madrid"]

    def test_film_hyphenated_stock(self):
        path = ROOT / "film/2025/nikon_f50/20250601_ilford-hp5_400/re001.jpg"
        result = parse_photo_path(path, ROOT)
        assert result.film_stock == "ilford-hp5"

    def test_film_unknown_stock_and_iso(self):
        path = ROOT / "film/2025/canon_prima_ii/20251202_x_0_paloma-salsa/re001.jpg"
        result = parse_photo_path(path, ROOT)
        assert result.film_stock is None
        assert result.film_iso is None
        assert result.roll_tags == ["paloma", "salsa"]

    def test_film_unknown_stock_only(self):
        path = ROOT / "film/2026/canon_prima_ii/20260331_x_100_vzla-2/re001.jpg"
        result = parse_photo_path(path, ROOT)
        assert result.film_stock is None
        assert result.film_iso == 100

    def test_film_too_few_parts(self):
        path = ROOT / "film/2025/nikon_f50/re001.jpg"
        with pytest.raises(ValueError, match="exactly 5 parts"):
            parse_photo_path(path, ROOT)
