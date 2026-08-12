"""Tests for ImageFolder: file discovery and the square-crop preprocessing."""

import numpy as np
import pytest
from PIL import Image

from dataset import ImageFolder


def _jpeg(path, w, h, mode="RGB"):
    arr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    if mode != "RGB":
        img = img.convert(mode)
    img.save(path, "JPEG")
    return path


# --- file discovery ---------------------------------------------------------


def test_skips_subdirectories(tmp_path):
    """Regression: os.listdir handed directories to rawpy -> LibRawIOError."""
    _jpeg(tmp_path / "a.jpg", 640, 480)
    (tmp_path / "subdir").mkdir()

    folder = ImageFolder(str(tmp_path))
    assert len(folder) == 1


def test_skips_non_image_files(tmp_path):
    """Regression: sidecar files (.txt, .xmp, .DS_Store) crashed the RAW decoder."""
    _jpeg(tmp_path / "a.jpg", 640, 480)
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "a.xmp").write_text("<xml/>")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")

    folder = ImageFolder(str(tmp_path))
    assert len(folder) == 1
    assert folder[0]["path"].endswith("a.jpg")


def test_finds_images_case_insensitively(tmp_path):
    _jpeg(tmp_path / "lower.jpg", 640, 480)
    _jpeg(tmp_path / "upper.JPG", 640, 480)
    _jpeg(tmp_path / "mixed.JpEg", 640, 480)

    assert len(ImageFolder(str(tmp_path))) == 3


def test_recursive_discovery(tmp_path):
    _jpeg(tmp_path / "top.jpg", 640, 480)
    nested = tmp_path / "2024" / "trip"
    nested.mkdir(parents=True)
    _jpeg(nested / "deep.jpg", 640, 480)

    assert len(ImageFolder(str(tmp_path), recursive=True)) == 2
    assert len(ImageFolder(str(tmp_path), recursive=False)) == 1


def test_accepts_a_single_file_path(tmp_path):
    path = _jpeg(tmp_path / "only.jpg", 640, 480)
    folder = ImageFolder(str(path))
    assert len(folder) == 1


def test_listing_is_deterministic(tmp_path):
    for name in ["c.jpg", "a.jpg", "b.jpg"]:
        _jpeg(tmp_path / name, 320, 240)
    names = [s["path"] for s in ImageFolder(str(tmp_path))]
    assert names == sorted(names)


def test_empty_folder_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No supported images"):
        ImageFolder(str(tmp_path))


# --- preprocessing ----------------------------------------------------------


@pytest.mark.parametrize(
    "w,h",
    [
        (640, 480),  # landscape
        (480, 640),  # portrait
        (512, 512),  # square - used to produce an empty array and crash cv2.resize
        (1000, 200),  # extreme panorama
        (200, 1000),  # extreme tall
        (257, 256),  # odd-by-one
    ],
)
def test_all_aspect_ratios_produce_valid_tensor(tmp_path, w, h):
    _jpeg(tmp_path / "img.jpg", w, h)
    sample = ImageFolder(str(tmp_path))[0]

    assert sample["img"].shape == (3, 256, 256)
    assert np.isfinite(sample["img"].numpy()).all()


def test_grayscale_image_is_handled(tmp_path):
    _jpeg(tmp_path / "gray.jpg", 640, 480, mode="L")
    sample = ImageFolder(str(tmp_path))[0]
    assert sample["img"].shape == (3, 256, 256)


def test_uniform_image_does_not_produce_nan(tmp_path):
    """min-max normalisation divides by (max-min); a flat image must not blow up."""
    Image.fromarray(np.full((480, 640, 3), 200, np.uint8)).save(tmp_path / "flat.jpg")
    sample = ImageFolder(str(tmp_path))[0]
    assert np.isfinite(sample["img"].numpy()).all()


def test_preprocessing_matches_training_pipeline(tmp_path):
    """The pretrained checkpoint is only valid for the exact training preprocessing:
    portrait orientation, centre crop to square, 256x256 cubic resize."""
    _jpeg(tmp_path / "img.jpg", 640, 480)
    sample = ImageFolder(str(tmp_path))[0]
    img = sample["img"].numpy()

    assert img.shape == (3, 256, 256)
    # ImageNet normalisation leaves values roughly in [-2.2, 2.7]
    assert img.min() > -3.0 and img.max() < 3.5


def test_sample_carries_path_and_raw_image(tmp_path):
    _jpeg(tmp_path / "img.jpg", 640, 480)
    sample = ImageFolder(str(tmp_path))[0]
    assert sample["path"].endswith("img.jpg")
    assert sample["raw_img"].ndim == 3
