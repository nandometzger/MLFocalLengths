<div align="center">

# 📷 MLFocalLengths

### Estimating the focal length of a single image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backbone: EfficientNet-B4](https://img.shields.io/badge/Backbone-EfficientNet--B4-4C9A2A.svg)](https://arxiv.org/abs/1905.11946)
[![Pretrained weights](https://img.shields.io/badge/Weights-auto--download-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/16Yf8dQrIAg-k8RKcy_chRsctrhQ4yzse/view?usp=share_link)

**Point it at a photo. Get the focal length back — no EXIF required.**

![Focal length predictions on a dolly-zoom sequence](img/stone_tagged2.gif)

<sub>Same subject size, changing focal length. The model reads the geometry of the scene and predicts the lens.<br>
Source sequence by Reddit user [u/scyshc](https://www.reddit.com/r/photography/comments/48l8uy/a_gif_showing_why_focal_length_matters/)</sub>

</div>

---

## Why this is interesting

The focal length a photo was taken with is often missing: stripped from images on the internet, never recorded on vintage film scans, or lost in a screenshot.

Recovering it from a **single** image is ill-posed. There is no depth cue to measure against, so the network has to lean on something else entirely: how big objects *ought* to be, how far away they *ought* to sit, and how perspective bends the scene at 16 mm versus 200 mm. In other words, guessing the lens requires understanding the scene.

This repository trains a CNN to do exactly that, and ships the pretrained model.

## Highlights

- 🎯 **16 mm mean absolute error** on a held-out set of real photographs
- 📐 Reports both the **35mm-equivalent** and the **physical** focal length, deriving the sensor crop factor from EXIF when the camera recorded it
- 🔍 **Reads any focal length metadata still in the file** and scores its own prediction against it
- 🤖 **JSON and CSV output**, non-interactive, for scripting and batch jobs
- 📦 **Weights download themselves** on first run — one command from clone to prediction

## Quick start

```bash
pip install -r requirements.txt
python predict.py img/stone
```

That is the whole setup. The pretrained weights (233 MB) are fetched automatically on first use and cached in `~/.cache/mlfocallengths/`.

```
frame_00_delay-0.03s.jpg     72.1 mm equiv    72.1 mm physical  crop 1x (assumed_full_frame)
frame_08_delay-0.03s.jpg     21.0 mm equiv    21.0 mm physical  crop 1x (assumed_full_frame)
```

On a photo that still has its EXIF, the model's prediction is scored against what the camera recorded:

```
$ python predict.py IMG_9010.CR2
IMG_9010.CR2     62.1 mm equiv    62.0 mm physical  crop 1.002x (exif_sensor_width)
                                  | EXIF 50.1 mm equiv [derived_from_physical], error +12.0 mm
```

### Common invocations

| Goal | Command |
|---|---|
| One image | `python predict.py photo.jpg` |
| A whole library | `python predict.py ~/Pictures --recursive` |
| Machine-readable output | `python predict.py ~/Pictures --json out.json --csv out.csv` |
| Annotated preview images | `python predict.py img/stone --save-demo demo/` |
| Known APS-C body | `python predict.py photo.jpg --crop-factor 1.5` |
| Tag the files themselves | `python predict.py ~/Pictures --write-exif` |

Run `python predict.py --help` for the full set.

## The two focal lengths

The network can only predict a **35mm-equivalent** focal length, because that is the one number comparable across sensor sizes — a 25 mm lens on Micro Four Thirds frames the same scene as a 50 mm lens on full frame, and the image alone cannot tell the two apart.

Turning that back into the **physical** focal length engraved on the lens needs the sensor's crop factor, which the pixels genuinely do not contain. It is taken from the first source available:

| Source | Meaning |
|---|---|
| `user_crop_factor` | You passed `--crop-factor` |
| `user_sensor_width` | You passed `--sensor-width` (mm) |
| `exif_35mm_ratio` | Camera recorded both focal lengths; their ratio is exact |
| `exif_sensor_width` | Derived from the EXIF `FocalPlane*` tags |
| `assumed_full_frame` | Nothing was available — physical equals equivalent, and this label says so |

Every result carries its source, so an assumed number is never mistaken for a measured one.

## Reading and writing EXIF

Whatever focal length metadata survives in the file is read out and reported alongside the prediction: physical focal length, 35mm equivalent, camera make and model, lens model. Where the camera recorded a physical focal length but no equivalent — common on older full-frame bodies — the equivalent is reconstructed from the measured crop factor and labelled `derived_from_physical`.

`--write-exif` records predictions back into JPEGs. It is **off by default**, and even when enabled it will not overwrite focal length tags the camera itself wrote — that metadata is provenance, and it is the ground truth this tool reports against. The prediction always lands in `UserComment`, clearly marked:

```
MLFocalLengths: 72.1mm 35mm-equivalent, 72.1mm physical (crop factor 1, assumed_full_frame) -- predicted, not measured
```

Pass `--overwrite-exif` to deliberately replace the camera's own tags. RAW files are read but never written to.

## Results

| | |
|---|---|
| Mean absolute error (hold-out) | **16 mm** (35 mm equivalent) |
| Backbone | EfficientNet-B4, ImageNet-initialised |
| Training data | ~15k personal photographs |
| Input resolution | 256 × 256, centre-cropped to square |
| Optimisation | Adam, L1 loss on log-focal-length |

## How it works

**Labels.** Focal lengths were read from the EXIF field `FocalLengthIn35mmFilm` and normalised to 35 mm equivalent with [Jeffrey Friedl's Lightroom plugin](http://regex.info/blog/lightroom-goodies/focal-length-sort), so that photos from different sensor sizes are directly comparable.

**Images.** Each photo is centre-cropped to a square, resized to 256 × 256, min-max normalised, then standardised with ImageNet statistics. The full set is cached into an HDF5 file so training does not re-decode RAWs every epoch. Splits are made **chronologically** rather than randomly, which keeps near-duplicate shots from the same session out of both train and test.

**Model.** An ImageNet-pretrained EfficientNet-B4 with a linear head that maps to a single scalar (`model.py`).

**Loss.** L1, but computed on `log(focal length)`. Focal length is perceptually multiplicative — the step from 16 mm to 24 mm matters far more than 180 mm to 188 mm — and the log transform makes the loss reflect that. A softplus keeps predictions positive before the log.

**Training.** Adam with a step LR schedule, gradient clipping, and horizontal/vertical flip augmentation. Metrics stream to [Weights & Biases](https://wandb.ai/) when you are logged in, and are silently skipped when you are not.

## Training it yourself

Training data is available upon request. Given a folder of photos with intact EXIF:

```bash
# 1. Build the HDF5 cache (keeps only images with a focal length tag)
python dataset.py --data-dir ~/Pictures/2022 --hdf5-path data/imgdataset4.h5

# 2. Train
python train.py --save-dir myoutdir --batch-size 64 --lr 0.0001 \
    --lr-step 4 --lr-gamma 0.9 --in_memory

# 3. Evaluate on the held-out split
python evaluate.py --checkpoint myoutdir/My/experiment_0/best_model.pth
```

Checkpoints land in `myoutdir/<dataset>/experiment_<n>/`. Add `--wandb disabled` to turn off experiment logging entirely, or `--wandb online` to force it. Run any script with `--help` for the full set of options.

## Tests

```bash
pip install pytest && python -m pytest tests/
```

47 tests covering the crop-factor maths, EXIF round-trips, and the image loader's edge cases (square, portrait, panoramic, grayscale, uniform images, and folders containing sidecar files). The end-to-end test runs only when the pretrained checkpoint is present locally.

## Repository layout

```
├── predict.py     Inference CLI: focal lengths, EXIF readout, JSON/CSV, previews
├── exif_tools.py  EXIF reading, crop-factor derivation, prediction write-back
├── model.py       EfficientNet-B4 regressor + log-space L1 loss
├── dataset.py     Image loading and preprocessing, HDF5 cache builder, splits
├── train.py       Training loop, optional W&B logging, checkpointing
├── evaluate.py    Evaluation on the held-out split
├── utils.py       Seeding, logging directories, CUDA helpers
├── tests/         pytest suite
└── img/           Demo sequences and example outputs
```

## Citation

If you use this work, please cite:

```bibtex
@misc{Metzger2023MLFocalLengths,
  author       = {Nando Metzger},
  title        = {MLFocalLengths: Estimating the Focal Length of a Single Image},
  year         = {2023},
  url          = {https://github.com/nandometzger/MLFocalLengths},
  note         = {GitHub repository}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Demo sequence courtesy of Reddit user [u/scyshc](https://www.reddit.com/r/photography/comments/48l8uy/a_gif_showing_why_focal_length_matters/). Focal length normalisation relies on [Jeffrey Friedl's Lightroom plugin](http://regex.info/blog/lightroom-goodies/focal-length-sort).
