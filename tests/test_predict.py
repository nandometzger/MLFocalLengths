"""Tests for the prediction CLI: argument handling, output files, and a real
end-to-end run when the pretrained checkpoint happens to be available."""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

import predict
from predict import CHECKPOINT_CACHE, build_parser, collate, describe, write_csv, write_json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _record(**overrides):
    record = {
        "path": "/photos/a.jpg",
        "predicted_focal_length_35mm_equiv_mm": 72.1,
        "predicted_focal_length_physical_mm": 48.1,
        "crop_factor": 1.5,
        "crop_factor_source": "exif_35mm_ratio",
        "exif": {
            "focal_length_mm": 50.0,
            "focal_length_35mm_equiv_mm": 75.0,
            "sensor_width_mm": None,
            "camera_make": "Canon",
            "camera_model": "EOS 5D",
            "lens_model": None,
            "user_comment": None,
        },
        "exif_reference_35mm_equiv_mm": 75.0,
        "exif_reference_source": "exif_35mm_tag",
        "error_vs_exif_35mm_mm": -2.9,
        "error_vs_exif_physical_mm": -1.9,
    }
    record.update(overrides)
    return record


# --- argument handling ------------------------------------------------------


def test_positional_input_is_accepted():
    args = build_parser().parse_args(["img/stone"])
    assert args.input == "img/stone"


def test_legacy_root_dir_flag_still_works():
    args = build_parser().parse_args(["--root_dir", "img/stone"])
    assert args.root_dir == "img/stone"


def test_save_demo_is_not_a_broken_bool():
    """Regression: --save_demo used type=bool, so `--save_demo False` meant True."""
    assert build_parser().parse_args(["x"]).save_demo is None
    assert build_parser().parse_args(["x", "--save-demo", "out"]).save_demo == "out"


def test_exif_writing_is_opt_in():
    """Modifying the user's photos in place must never be the default."""
    assert build_parser().parse_args(["x"]).write_exif is False


def test_missing_input_is_a_clean_error():
    args = build_parser().parse_args([])
    with pytest.raises(SystemExit, match="give an image or a directory"):
        predict.run(args)


def test_cuda_request_without_gpu_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="no CUDA device"):
        predict.resolve_device("cuda")


def test_missing_checkpoint_is_a_clean_error():
    with pytest.raises(SystemExit, match="no checkpoint at"):
        predict.ensure_checkpoint("/nonexistent/model.pth")


# --- batching and formatting ------------------------------------------------


def test_collate_keeps_paths_as_plain_strings():
    """Default collation mangles strings; paths must survive intact."""
    batch = collate([
        {"img": torch.zeros(3, 8, 8), "path": "/photos/a.jpg"},
        {"img": torch.zeros(3, 8, 8), "path": "/photos/b.jpg"},
    ])
    assert batch["path"] == ["/photos/a.jpg", "/photos/b.jpg"]
    assert batch["img"].shape == (2, 3, 8, 8)


def test_describe_reports_both_focal_lengths():
    line = describe(_record())
    assert "72.1 mm equiv" in line
    assert "48.1 mm physical" in line
    assert "-2.9 mm" in line


def test_describe_without_exif_omits_the_comparison():
    line = describe(_record(exif_reference_35mm_equiv_mm=None, error_vs_exif_35mm_mm=None))
    assert "error" not in line


# --- output files -----------------------------------------------------------


def test_json_output_is_machine_readable(tmp_path):
    out = tmp_path / "r.json"
    summary = {"images": 1, "images_with_exif_focal_length": 1,
               "mean_absolute_error_vs_exif_mm": 2.9, "images_tagged": 0}
    write_json(str(out), [_record()], "ckpt.pth", summary)

    payload = json.loads(out.read_text())
    assert payload["model"]["predicts"] == "35mm_equivalent_focal_length_mm"
    assert payload["summary"]["images"] == 1
    assert payload["predictions"][0]["predicted_focal_length_physical_mm"] == 48.1


def test_csv_output_has_both_focal_lengths(tmp_path):
    out = tmp_path / "r.csv"
    write_csv(str(out), [_record()])

    rows = list(csv.DictReader(out.open()))
    assert rows[0]["predicted_focal_length_35mm_equiv_mm"] == "72.1"
    assert rows[0]["predicted_focal_length_physical_mm"] == "48.1"
    assert rows[0]["camera_model"] == "EOS 5D"


# --- end to end -------------------------------------------------------------


def _checkpoint():
    for candidate in [os.environ.get("MLFL_CHECKPOINT"), CHECKPOINT_CACHE]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


@pytest.mark.skipif(_checkpoint() is None, reason="pretrained checkpoint not available")
def test_end_to_end_prediction_on_real_images(tmp_path):
    """Run the actual CLI the way the README documents it."""
    for i in range(2):
        arr = np.random.default_rng(i).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img{i}.jpg")

    out = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, "predict.py", str(tmp_path), "--checkpoint", _checkpoint(),
         "--json", str(out), "--num-workers", "0", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.read_text())
    assert payload["summary"]["images"] == 2
    for record in payload["predictions"]:
        equiv = record["predicted_focal_length_35mm_equiv_mm"]
        assert 1.0 < equiv < 2000.0, f"implausible focal length: {equiv}"
        assert record["predicted_focal_length_physical_mm"] > 0
