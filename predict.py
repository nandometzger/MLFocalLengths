"""Predict the focal length of photographs.

The network outputs a 35mm-equivalent focal length. Where the image still carries
EXIF metadata, that is read out alongside the prediction -- both as a sanity check
and to recover the sensor crop factor, which is what turns the equivalent focal
length into the physical focal length that was on the lens.

    python predict.py img/stone
    python predict.py photo.jpg --json results.json
    python predict.py ~/Pictures --recursive --csv results.csv
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")  # render demo images without needing a display

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RAW_EXTENSIONS, ImageFolder
from exif_tools import (
    equivalent_to_physical,
    read_exif_focal_info,
    reference_equivalent,
    resolve_crop_factor,
    write_prediction_exif,
)
from model import CNN

CHECKPOINT_URL = "https://drive.google.com/uc?id=16Yf8dQrIAg-k8RKcy_chRsctrhQ4yzse"
CHECKPOINT_CACHE = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "mlfocallengths",
    "best_model.pth",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate the focal length of one image or a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="Image file or directory of images")
    parser.add_argument("--root_dir", dest="root_dir", help="Deprecated alias for the positional input")

    parser.add_argument("--checkpoint", help="Model weights (downloaded and cached on first use)")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories too")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    sensor = parser.add_argument_group("sensor geometry (for the physical focal length)")
    sensor.add_argument("--crop-factor", type=float, help="Override the crop factor, e.g. 1.5 for APS-C")
    sensor.add_argument("--sensor-width", type=float, metavar="MM", help="Sensor width in mm, e.g. 23.5")

    out = parser.add_argument_group("output")
    out.add_argument("--json", metavar="PATH", help="Write results as JSON")
    out.add_argument("--csv", metavar="PATH", help="Write results as CSV")
    out.add_argument("--save-demo", metavar="DIR", help="Render annotated preview images into DIR")
    out.add_argument("--quiet", action="store_true", help="Only report errors")

    exif = parser.add_argument_group("EXIF write-back (off by default; modifies files in place)")
    exif.add_argument("--write-exif", action="store_true",
                      help="Record the prediction in each JPEG's EXIF UserComment")
    exif.add_argument("--overwrite-exif", action="store_true",
                      help="With --write-exif, also replace focal length tags the camera recorded")
    return parser


def resolve_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but no CUDA device is available")
    return torch.device(choice)


def _report_download(block, block_size, total):
    if total <= 0:
        return
    done = min(block * block_size, total)
    pct = 100.0 * done / total
    sys.stderr.write(f"\r  {pct:5.1f}%  ({done / 1e6:6.1f} / {total / 1e6:.1f} MB)")
    sys.stderr.flush()


def ensure_checkpoint(path=None):
    """Return a local checkpoint path, downloading the pretrained weights if needed."""
    if path:
        if not os.path.isfile(path):
            raise SystemExit(f"error: no checkpoint at '{path}'")
        return path

    if os.path.isfile(CHECKPOINT_CACHE):
        return CHECKPOINT_CACHE

    os.makedirs(os.path.dirname(CHECKPOINT_CACHE), exist_ok=True)
    print(f"Downloading pretrained weights (233 MB) to {CHECKPOINT_CACHE}", file=sys.stderr)
    partial = CHECKPOINT_CACHE + ".part"
    try:
        try:
            import gdown

            gdown.download(CHECKPOINT_URL, partial, quiet=False)
        except ImportError:
            urllib.request.urlretrieve(CHECKPOINT_URL, partial, reporthook=_report_download)
            sys.stderr.write("\n")
        os.replace(partial, CHECKPOINT_CACHE)
    except Exception as error:
        if os.path.exists(partial):
            os.remove(partial)
        raise SystemExit(
            f"error: could not download the weights ({error}).\n"
            f"Download them manually from {CHECKPOINT_URL}\n"
            f"and pass --checkpoint <path>, or place the file at {CHECKPOINT_CACHE}"
        )
    return CHECKPOINT_CACHE


def load_model(checkpoint_path, device):
    model = CNN()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Older checkpoints pickle the optimizer state alongside the weights.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device).eval()


def collate(batch):
    """Keep paths and raw images as plain Python; only the tensors get stacked."""
    return {
        "img": torch.stack([item["img"] for item in batch]),
        "path": [item["path"] for item in batch],
        "raw_img": [item.get("raw_img") for item in batch],
    }


def describe(record):
    """One human-readable line per image."""
    line = (
        f"{os.path.basename(record['path']):<34.34} "
        f"{record['predicted_focal_length_35mm_equiv_mm']:6.1f} mm equiv"
        f"  {record['predicted_focal_length_physical_mm']:6.1f} mm physical"
        f"  crop {record['crop_factor']:.4g}x ({record['crop_factor_source']})"
    )
    reference = record["exif_reference_35mm_equiv_mm"]
    if reference is not None:
        line += (
            f"  | EXIF {reference:.1f} mm equiv"
            f" [{record['exif_reference_source']}], error {record['error_vs_exif_35mm_mm']:+.1f} mm"
        )
    return line


def save_demo_image(record, raw_img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    try:
        plt.imshow(raw_img)
        plt.axis("off")
        plt.title(
            f"Predicted: {record['predicted_focal_length_35mm_equiv_mm']:.1f} mm (35mm equiv.)\n"
            f"{record['predicted_focal_length_physical_mm']:.1f} mm physical "
            f"at crop {record['crop_factor']:.3g}x"
        )
        out_path = os.path.join(out_dir, "tagged_" + os.path.basename(record["path"]))
        fig.savefig(out_path, bbox_inches="tight", dpi=110)
    finally:
        plt.close(fig)  # closing matters: one figure per image otherwise leaks memory


def write_json(path, records, checkpoint, summary):
    payload = {
        "model": {
            "checkpoint": checkpoint,
            "architecture": "efficientnet_b4",
            "predicts": "35mm_equivalent_focal_length_mm",
        },
        "summary": summary,
        "predictions": records,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, records):
    import csv

    columns = [
        "path",
        "predicted_focal_length_35mm_equiv_mm",
        "predicted_focal_length_physical_mm",
        "crop_factor",
        "crop_factor_source",
        "exif_focal_length_mm",
        "exif_focal_length_35mm_equiv_mm",
        "exif_reference_35mm_equiv_mm",
        "exif_reference_source",
        "error_vs_exif_35mm_mm",
        "error_vs_exif_physical_mm",
        "camera_make",
        "camera_model",
        "lens_model",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            exif = record["exif"]
            writer.writerow({
                "path": record["path"],
                "predicted_focal_length_35mm_equiv_mm": round(record["predicted_focal_length_35mm_equiv_mm"], 2),
                "predicted_focal_length_physical_mm": round(record["predicted_focal_length_physical_mm"], 2),
                "crop_factor": round(record["crop_factor"], 4),
                "crop_factor_source": record["crop_factor_source"],
                "exif_focal_length_mm": exif["focal_length_mm"],
                "exif_focal_length_35mm_equiv_mm": exif["focal_length_35mm_equiv_mm"],
                "exif_reference_35mm_equiv_mm": record["exif_reference_35mm_equiv_mm"],
                "exif_reference_source": record["exif_reference_source"],
                "error_vs_exif_35mm_mm": record["error_vs_exif_35mm_mm"],
                "error_vs_exif_physical_mm": record["error_vs_exif_physical_mm"],
                "camera_make": exif["camera_make"],
                "camera_model": exif["camera_model"],
                "lens_model": exif["lens_model"],
            })


@torch.no_grad()
def run(args):
    source = args.input or args.root_dir
    if not source:
        raise SystemExit("error: give an image or a directory, e.g. `python predict.py img/stone`")

    device = resolve_device(args.device)
    checkpoint_path = ensure_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, device)

    need_raw = args.save_demo is not None
    try:
        dataset = ImageFolder(source, recursive=args.recursive, return_raw=need_raw)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}")

    loader_kwargs = {}
    if args.num_workers > 0 and any(
        os.path.splitext(f)[1].lower() in RAW_EXTENSIONS for f in dataset.files
    ):
        # rawpy's OpenMP build can deadlock in fork()ed dataloader workers.
        loader_kwargs["multiprocessing_context"] = "forkserver"

    loader = DataLoader(
        dataset,
        batch_size=1 if need_raw else args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )

    if not args.quiet:
        print(f"{len(dataset)} image(s) on {device}, weights: {checkpoint_path}\n")

    records, tagged = [], 0
    progress = tqdm(total=len(dataset), disable=args.quiet or len(dataset) < 2,
                    unit="img", desc="Predicting", leave=False)

    for batch in loader:
        predictions = model({"img": batch["img"].to(device)})[:, 0].cpu().tolist()

        for path, equiv, raw_img in zip(batch["path"], predictions, batch["raw_img"]):
            info = read_exif_focal_info(path)
            crop, crop_source = resolve_crop_factor(info, args.crop_factor, args.sensor_width)
            physical = equivalent_to_physical(equiv, crop)

            reference, reference_source = reference_equivalent(info, crop, crop_source)
            record = {
                "path": path,
                "predicted_focal_length_35mm_equiv_mm": round(float(equiv), 2),
                "predicted_focal_length_physical_mm": round(float(physical), 2),
                "crop_factor": round(float(crop), 4),
                "crop_factor_source": crop_source,
                "exif": info.to_dict(),
                "exif_reference_35mm_equiv_mm": round(reference, 2) if reference else None,
                "exif_reference_source": reference_source,
                "error_vs_exif_35mm_mm": round(float(equiv) - reference, 2) if reference else None,
                "error_vs_exif_physical_mm": (
                    round(float(physical) - float(info.focal_length_mm), 2) if info.focal_length_mm else None
                ),
            }

            if args.write_exif:
                tagged += write_prediction_exif(
                    path, equiv, physical,
                    overwrite=args.overwrite_exif,
                    crop_factor=crop, crop_factor_source=crop_source,
                )
            if args.save_demo and raw_img is not None:
                save_demo_image(record, raw_img, args.save_demo)

            records.append(record)
            progress.update(1)
            if not args.quiet:
                progress.write(describe(record))

    progress.close()

    errors = [abs(r["error_vs_exif_35mm_mm"]) for r in records if r["error_vs_exif_35mm_mm"] is not None]
    summary = {
        "images": len(records),
        "images_with_exif_focal_length": len(errors),
        "mean_absolute_error_vs_exif_mm": round(sum(errors) / len(errors), 2) if errors else None,
        "images_tagged": tagged if args.write_exif else 0,
    }

    if args.json:
        write_json(args.json, records, checkpoint_path, summary)
    if args.csv:
        write_csv(args.csv, records)

    if not args.quiet:
        print(f"\n{summary['images']} image(s) processed.")
        if errors:
            print(f"Mean absolute error against EXIF ({len(errors)} image(s)): "
                  f"{summary['mean_absolute_error_vs_exif_mm']:.1f} mm")
        if args.write_exif:
            print(f"EXIF updated on {tagged} JPEG(s).")
        for label, path in (("JSON", args.json), ("CSV", args.csv), ("Demo images", args.save_demo)):
            if path:
                print(f"{label}: {path}")

    return records


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    started = time.time()
    run(arguments)
    if not arguments.quiet:
        print(f"Done in {time.time() - started:.1f}s")
