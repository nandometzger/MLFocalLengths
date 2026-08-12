"""Tests for EXIF reading, crop-factor derivation and prediction write-back."""

import numpy as np
import piexif
import pytest
from PIL import Image

from exif_tools import (
    ExifFocalInfo,
    equivalent_to_physical,
    physical_to_equivalent,
    read_exif_focal_info,
    resolve_crop_factor,
    write_prediction_exif,
)


def _write_jpeg(path, exif_bytes=None, size=(640, 480)):
    img = Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8))
    if exif_bytes is None:
        img.save(path, "JPEG")
    else:
        img.save(path, "JPEG", exif=exif_bytes)
    return str(path)


def _exif_bytes(zeroth=None, exif=None):
    return piexif.dump({"0th": zeroth or {}, "Exif": exif or {}, "GPS": {}, "1st": {}, "thumbnail": None})


# --- crop factor math -------------------------------------------------------


def test_crop_factor_from_35mm_ratio():
    """A 50mm lens reported as 75mm equivalent means a 1.5x crop sensor."""
    info = ExifFocalInfo(focal_length_mm=50.0, focal_length_35mm_equiv_mm=75.0)
    crop, source = resolve_crop_factor(info)
    assert crop == pytest.approx(1.5)
    assert source == "exif_35mm_ratio"


def test_crop_factor_from_focal_plane_resolution():
    """APS-C: 6000px wide at 4233 px/inch -> ~36mm/1.6 sensor width."""
    info = ExifFocalInfo(sensor_width_mm=22.3)
    crop, source = resolve_crop_factor(info)
    assert crop == pytest.approx(36.0 / 22.3, rel=1e-6)
    assert source == "exif_sensor_width"


def test_explicit_crop_factor_wins_over_exif():
    info = ExifFocalInfo(focal_length_mm=50.0, focal_length_35mm_equiv_mm=75.0)
    crop, source = resolve_crop_factor(info, crop_factor=2.0)
    assert crop == pytest.approx(2.0)
    assert source == "user_crop_factor"


def test_explicit_sensor_width_wins_over_exif():
    info = ExifFocalInfo(focal_length_mm=50.0, focal_length_35mm_equiv_mm=75.0)
    crop, source = resolve_crop_factor(info, sensor_width_mm=18.0)
    assert crop == pytest.approx(2.0)
    assert source == "user_sensor_width"


def test_crop_factor_defaults_to_full_frame_and_says_so():
    crop, source = resolve_crop_factor(ExifFocalInfo())
    assert crop == pytest.approx(1.0)
    assert source == "assumed_full_frame"


def test_zero_focal_length_does_not_divide_by_zero():
    info = ExifFocalInfo(focal_length_mm=0.0, focal_length_35mm_equiv_mm=75.0)
    crop, source = resolve_crop_factor(info)
    assert crop == pytest.approx(1.0)
    assert source == "assumed_full_frame"


def test_equivalent_physical_roundtrip():
    assert equivalent_to_physical(75.0, 1.5) == pytest.approx(50.0)
    assert physical_to_equivalent(50.0, 1.5) == pytest.approx(75.0)


# --- EXIF reading -----------------------------------------------------------


def test_read_exif_returns_focal_lengths(tmp_path):
    path = _write_jpeg(
        tmp_path / "shot.jpg",
        _exif_bytes(
            zeroth={piexif.ImageIFD.Make: b"Canon", piexif.ImageIFD.Model: b"EOS 80D"},
            exif={
                piexif.ExifIFD.FocalLength: (50, 1),
                piexif.ExifIFD.FocalLengthIn35mmFilm: 80,
                piexif.ExifIFD.LensModel: b"EF-S 18-55mm",
            },
        ),
    )
    info = read_exif_focal_info(path)
    assert info.focal_length_mm == pytest.approx(50.0)
    assert info.focal_length_35mm_equiv_mm == pytest.approx(80.0)
    assert info.camera_make == "Canon"
    assert info.camera_model == "EOS 80D"
    assert info.lens_model == "EF-S 18-55mm"
    assert info.has_any() is True


def test_read_exif_on_image_without_exif(tmp_path):
    info = read_exif_focal_info(_write_jpeg(tmp_path / "bare.jpg"))
    assert info.focal_length_mm is None
    assert info.focal_length_35mm_equiv_mm is None
    assert info.has_any() is False


def test_read_exif_derives_sensor_width_from_focal_plane(tmp_path):
    """FocalPlaneXResolution in inches -> sensor width in mm."""
    path = _write_jpeg(
        tmp_path / "fp.jpg",
        _exif_bytes(
            exif={
                piexif.ExifIFD.FocalPlaneXResolution: (1524000, 223),  # 6000px / 22.3mm, in px/inch
                piexif.ExifIFD.FocalPlaneResolutionUnit: 2,  # inch
                piexif.ExifIFD.PixelXDimension: 6000,
            }
        ),
        size=(6000, 4000),
    )
    info = read_exif_focal_info(path)
    assert info.sensor_width_mm == pytest.approx(22.3, abs=0.1)


def test_read_exif_on_missing_file_does_not_raise(tmp_path):
    info = read_exif_focal_info(str(tmp_path / "nope.jpg"))
    assert info.has_any() is False


def test_read_exif_on_non_image_does_not_raise(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("definitely not a jpeg")
    assert read_exif_focal_info(str(junk)).has_any() is False


# --- EXIF writing (the original silent-failure bug) -------------------------


def test_written_prediction_survives_read_back(tmp_path):
    """Regression: the old code stuffed a bogus key into the dict and piexif dropped it."""
    path = _write_jpeg(tmp_path / "tag.jpg")
    assert write_prediction_exif(path, equiv_mm=72.1, physical_mm=48.1) is True

    info = read_exif_focal_info(path)
    assert info.focal_length_35mm_equiv_mm == pytest.approx(72.0, abs=1.0)
    assert info.focal_length_mm == pytest.approx(48.1, abs=0.1)
    assert "MLFocalLengths" in (info.user_comment or "")


def test_write_does_not_clobber_real_camera_exif(tmp_path):
    """Genuine camera metadata is provenance; never silently overwrite it."""
    path = _write_jpeg(
        tmp_path / "real.jpg",
        _exif_bytes(exif={piexif.ExifIFD.FocalLength: (35, 1), piexif.ExifIFD.FocalLengthIn35mmFilm: 52}),
    )
    write_prediction_exif(path, equiv_mm=100.0, physical_mm=66.0)

    info = read_exif_focal_info(path)
    assert info.focal_length_35mm_equiv_mm == pytest.approx(52.0)
    assert info.focal_length_mm == pytest.approx(35.0)
    assert "MLFocalLengths" in (info.user_comment or "")  # prediction still recorded


def test_write_with_overwrite_replaces_camera_exif(tmp_path):
    path = _write_jpeg(
        tmp_path / "real.jpg",
        _exif_bytes(exif={piexif.ExifIFD.FocalLength: (35, 1), piexif.ExifIFD.FocalLengthIn35mmFilm: 52}),
    )
    write_prediction_exif(path, equiv_mm=100.0, physical_mm=66.0, overwrite=True)

    info = read_exif_focal_info(path)
    assert info.focal_length_35mm_equiv_mm == pytest.approx(100.0)
    assert info.focal_length_mm == pytest.approx(66.0, abs=0.1)


def test_write_on_unsupported_format_returns_false(tmp_path):
    png = tmp_path / "img.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(png)
    assert write_prediction_exif(str(png), equiv_mm=50.0, physical_mm=50.0) is False


def test_write_preserves_image_pixels(tmp_path):
    path = _write_jpeg(tmp_path / "pixels.jpg")
    before = np.array(Image.open(path))
    write_prediction_exif(path, equiv_mm=50.0, physical_mm=50.0)
    after = np.array(Image.open(path))
    assert np.array_equal(before, after)
