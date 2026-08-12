"""EXIF helpers for focal length metadata.

The network predicts a **35mm-equivalent** focal length, because that is the only
focal length that is comparable across sensor sizes. Turning that back into the
**physical** focal length that was on the lens barrel needs the crop factor of the
sensor, which the image itself cannot tell us -- so we take it from EXIF when the
camera recorded it, or from the user, and otherwise say we assumed full frame.

Everything here degrades gracefully: unreadable or absent metadata yields an empty
``ExifFocalInfo`` rather than an exception, because a missing EXIF block is the
normal case for the images this project targets.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Optional, Tuple

import piexif

# Long edge of a 35mm full-frame sensor. Crop factor is defined here relative to
# sensor *width*, which is the convention camera makers use for FocalPlane tags.
FULL_FRAME_WIDTH_MM = 36.0

JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}

# EXIF FocalPlaneResolutionUnit -> millimetres per unit
_RESOLUTION_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}

_USER_COMMENT_PREFIXES = (b"ASCII\x00\x00\x00", b"UNICODE\x00", b"JIS\x00\x00\x00\x00\x00", b"\x00" * 8)

MARKER = "MLFocalLengths"


@dataclass
class ExifFocalInfo:
    """Focal-length-related metadata recovered from an image, all fields optional."""

    focal_length_mm: Optional[float] = None
    focal_length_35mm_equiv_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    user_comment: Optional[str] = None

    def has_any(self) -> bool:
        """True when the file carried any usable metadata at all."""
        return any(v is not None for v in asdict(self).values())

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# crop factor
# --------------------------------------------------------------------------


def resolve_crop_factor(
    info: ExifFocalInfo,
    crop_factor: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
) -> Tuple[float, str]:
    """Work out the sensor crop factor and report where the number came from.

    Priority: explicit user input, then what the camera recorded, then a
    full-frame assumption. The source string matters -- it is the difference
    between a measured physical focal length and a guessed one.
    """
    if crop_factor is not None and crop_factor > 0:
        return float(crop_factor), "user_crop_factor"

    if sensor_width_mm is not None and sensor_width_mm > 0:
        return FULL_FRAME_WIDTH_MM / float(sensor_width_mm), "user_sensor_width"

    physical, equiv = info.focal_length_mm, info.focal_length_35mm_equiv_mm
    if physical and equiv and physical > 0 and equiv > 0:
        return float(equiv) / float(physical), "exif_35mm_ratio"

    if info.sensor_width_mm and info.sensor_width_mm > 0:
        return FULL_FRAME_WIDTH_MM / float(info.sensor_width_mm), "exif_sensor_width"

    return 1.0, "assumed_full_frame"


def reference_equivalent(info: ExifFocalInfo, crop_factor: float, crop_factor_source: str):
    """Best available EXIF ground truth for the 35mm-equivalent focal length.

    Cameras that predate the ``FocalLengthIn35mmFilm`` tag (or full-frame bodies
    that see no need for it) still record the physical focal length. Combined with
    a crop factor we actually measured -- never an assumed one -- that recovers a
    usable reference. Returns ``(value, source)`` with ``(None, None)`` when the
    file gives us nothing to compare against.
    """
    if info.focal_length_35mm_equiv_mm:
        return float(info.focal_length_35mm_equiv_mm), "exif_35mm_tag"

    if info.focal_length_mm and crop_factor_source != "assumed_full_frame":
        return physical_to_equivalent(info.focal_length_mm, crop_factor), "derived_from_physical"

    return None, None


def equivalent_to_physical(equiv_mm: float, crop_factor: float) -> float:
    """35mm-equivalent focal length -> physical focal length on the lens."""
    if not crop_factor or crop_factor <= 0:
        return float(equiv_mm)
    return float(equiv_mm) / float(crop_factor)


def physical_to_equivalent(physical_mm: float, crop_factor: float) -> float:
    """Physical focal length -> 35mm equivalent."""
    return float(physical_mm) * float(crop_factor)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _as_float(value) -> Optional[float]:
    """EXIF numbers arrive as ints, (num, den) rationals or exifread Ratio objects."""
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        if isinstance(value, Fraction):
            return float(value)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _as_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text or None


def _decode_user_comment(raw) -> Optional[str]:
    """UserComment carries an 8-byte character-set prefix that must be stripped."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        for prefix in _USER_COMMENT_PREFIXES:
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]
                break
        return _as_text(raw.decode("utf-8", errors="replace"))
    return _as_text(raw)


def _sensor_width_from_focal_plane(x_resolution, unit, pixel_width) -> Optional[float]:
    """Derive physical sensor width from the FocalPlane* tags.

    sensor_width_mm = pixels / (pixels per unit) * (mm per unit)
    """
    res = _as_float(x_resolution)
    px = _as_float(pixel_width)
    mm_per_unit = _RESOLUTION_UNIT_MM.get(int(_as_float(unit) or 0))
    if not res or not px or not mm_per_unit or res <= 0:
        return None
    width = px / res * mm_per_unit
    # Sanity bound: real sensors sit between a phone camera and large format.
    return width if 1.0 < width < 200.0 else None


def _read_with_piexif(path: str) -> ExifFocalInfo:
    exif_dict = piexif.load(path)
    zeroth, exif = exif_dict.get("0th", {}), exif_dict.get("Exif", {})

    return ExifFocalInfo(
        focal_length_mm=_as_float(exif.get(piexif.ExifIFD.FocalLength)),
        focal_length_35mm_equiv_mm=_as_float(exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)),
        sensor_width_mm=_sensor_width_from_focal_plane(
            exif.get(piexif.ExifIFD.FocalPlaneXResolution),
            exif.get(piexif.ExifIFD.FocalPlaneResolutionUnit),
            exif.get(piexif.ExifIFD.PixelXDimension),
        ),
        camera_make=_as_text(zeroth.get(piexif.ImageIFD.Make)),
        camera_model=_as_text(zeroth.get(piexif.ImageIFD.Model)),
        lens_model=_as_text(exif.get(piexif.ExifIFD.LensModel)),
        user_comment=_decode_user_comment(exif.get(piexif.ExifIFD.UserComment)),
    )


def _read_with_exifread(path: str) -> ExifFocalInfo:
    """Fallback reader -- handles RAW formats (CR2, NEF, ARW, ...) that piexif cannot."""
    import exifread

    with open(path, "rb") as handle:
        tags = exifread.process_file(handle, details=False)

    def tag(name):
        entry = tags.get(name)
        return entry.values if entry is not None else None

    def first(name):
        values = tag(name)
        if isinstance(values, (list, tuple)) and values:
            return values[0]
        return values

    return ExifFocalInfo(
        focal_length_mm=_as_float(first("EXIF FocalLength")),
        focal_length_35mm_equiv_mm=_as_float(first("EXIF FocalLengthIn35mmFilm")),
        sensor_width_mm=_sensor_width_from_focal_plane(
            first("EXIF FocalPlaneXResolution"),
            first("EXIF FocalPlaneResolutionUnit"),
            first("EXIF ExifImageWidth"),
        ),
        camera_make=_as_text(tag("Image Make")),
        camera_model=_as_text(tag("Image Model")),
        lens_model=_as_text(tag("EXIF LensModel")),
        user_comment=_decode_user_comment(tag("EXIF UserComment")),
    )


def read_exif_focal_info(path: str) -> ExifFocalInfo:
    """Read focal length metadata from an image of any supported format.

    Never raises: an unreadable, missing or metadata-free file yields an empty
    ``ExifFocalInfo``.
    """
    if not os.path.isfile(path):
        return ExifFocalInfo()

    extension = os.path.splitext(path)[1].lower()
    readers = [_read_with_piexif, _read_with_exifread]
    if extension not in JPEG_EXTENSIONS:
        readers.reverse()  # RAW and friends: try exifread first

    for reader in readers:
        try:
            info = reader(path)
        except Exception:
            continue
        if info.has_any():
            return info

    return ExifFocalInfo()


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _to_rational(value: float, precision: int = 10) -> Tuple[int, int]:
    return int(round(float(value) * precision)), precision


def write_prediction_exif(
    path: str,
    equiv_mm: float,
    physical_mm: float,
    overwrite: bool = False,
    crop_factor: Optional[float] = None,
    crop_factor_source: Optional[str] = None,
) -> bool:
    """Write a prediction into an image's EXIF block. Returns True when written.

    The prediction always lands in ``UserComment``, tagged with the model name so
    it can never be mistaken for camera-recorded data. The standard
    ``FocalLength`` / ``FocalLengthIn35mmFilm`` tags are only filled in when the
    file does not already carry them -- genuine camera metadata is provenance and
    overwriting it silently would destroy the very ground truth this tool reports
    against. Pass ``overwrite=True`` to replace it deliberately.

    Only JPEG is supported; writing EXIF into a RAW file risks corrupting it, so
    other formats return False and are left untouched.
    """
    if os.path.splitext(path)[1].lower() not in JPEG_EXTENSIONS:
        return False

    try:
        exif_dict = piexif.load(path)
    except Exception:
        return False

    exif = exif_dict.setdefault("Exif", {})

    details = f"{MARKER}: {equiv_mm:.1f}mm 35mm-equivalent, {physical_mm:.1f}mm physical"
    if crop_factor is not None:
        details += f" (crop factor {crop_factor:.3g}"
        details += f", {crop_factor_source})" if crop_factor_source else ")"
    details += " -- predicted, not measured"
    exif[piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + details.encode("ascii", errors="replace")

    if overwrite or piexif.ExifIFD.FocalLengthIn35mmFilm not in exif:
        exif[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(equiv_mm))
    if overwrite or piexif.ExifIFD.FocalLength not in exif:
        exif[piexif.ExifIFD.FocalLength] = _to_rational(physical_mm)

    try:
        # A malformed thumbnail in the source file breaks dump(); dropping it is
        # preferable to refusing to tag the image.
        try:
            exif_bytes = piexif.dump(exif_dict)
        except Exception:
            exif_dict["thumbnail"] = None
            exif_dict["1st"] = {}
            exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)
    except Exception:
        return False

    return True
