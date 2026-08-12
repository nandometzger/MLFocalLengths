<div align="center">

# 📷 MLFocalLengths

### Estimating the focal length of a single image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backbone: EfficientNet-B4](https://img.shields.io/badge/Backbone-EfficientNet--B4-4C9A2A.svg)](https://arxiv.org/abs/1905.11946)
[![Pretrained weights](https://img.shields.io/badge/Weights-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/16Yf8dQrIAg-k8RKcy_chRsctrhQ4yzse/view?usp=share_link)

**Point it at a photo. Get the 35 mm-equivalent focal length back — no EXIF required.**

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
- 🧠 **EfficientNet-B4** regression head trained with an **L1 loss in log-space** — so a 10 mm mistake at wide angle counts more than at telephoto
- 🏷️ **Writes predictions straight into the image EXIF**, so results travel with the file
- 🖼️ **RAW and JPEG** input, with optional annotated preview images
- 📦 **Pretrained weights included** — run inference without training anything

## Results

| | |
|---|---|
| Mean absolute error (hold-out) | **16 mm** (35 mm equivalent) |
| Backbone | EfficientNet-B4, ImageNet-initialised |
| Training data | ~15k personal photographs |
| Input resolution | 256 × 256, centre-cropped to square |
| Optimisation | Adam, L1 loss on log-focal-length |

## Quick start

### 1. Set up the environment

`requirements.txt` is a conda specification (linux-64):

```bash
conda create --name focallengths --file requirements.txt
conda activate focallengths
```

### 2. Get the pretrained model

Download the checkpoint from [Google Drive](https://drive.google.com/file/d/16Yf8dQrIAg-k8RKcy_chRsctrhQ4yzse/view?usp=share_link) and put it somewhere reachable, e.g. `checkpoints/best_model.pth`.

### 3. Predict

```bash
python predict.py \
    --checkpoint checkpoints/best_model.pth \
    --root_dir img/stone \
    --save_demo True
```

Every image in `--root_dir` is processed, and for each one you get:

```
img/stone/frame_00.jpg  Predicted  72.2mm
```

- **JPEGs** additionally receive an `MLFocalLength` entry written into their EXIF block.
- **`--save_demo True`** renders annotated copies next to the input folder, in `<folder>_tagged/`.
- RAW files (e.g. `.CR2`) are decoded via `rawpy`; EXIF tagging is JPEG-only.

Runs on GPU when one is available, otherwise on CPU.

## How it works

**Labels.** Focal lengths were read from the EXIF field `FocalLengthIn35mmFilm` and normalised to 35 mm equivalent with [Jeffrey Friedl's Lightroom plugin](http://regex.info/blog/lightroom-goodies/focal-length-sort), so that photos from different sensor sizes are directly comparable.

**Images.** Each photo is centre-cropped to a square, resized to 256 × 256, min-max normalised, then standardised with ImageNet statistics. The full set is cached into an HDF5 file so training does not re-decode RAWs every epoch. Splits are made **chronologically** rather than randomly, which keeps near-duplicate shots from the same session out of both train and test.

**Model.** An ImageNet-pretrained EfficientNet-B4 with a linear head that maps to a single scalar (`model.py`).

**Loss.** L1, but computed on `log(focal length)`. Focal length is perceptually multiplicative — the step from 16 mm to 24 mm matters far more than 180 mm to 188 mm — and the log transform makes the loss reflect that. A softplus keeps predictions positive before the log.

**Training.** Adam with a step LR schedule, gradient clipping, and horizontal/vertical flip augmentation. Metrics stream to [Weights & Biases](https://wandb.ai/) under the project `focallengths`.

## Training it yourself

Training data is available upon request. Once you have a folder of photos with intact EXIF:

**Build the HDF5 dataset** (paths are configured at the bottom of `dataset.py`):

```bash
python dataset.py
```

**Train:**

```bash
python train.py \
    --save-dir myoutdir \
    --batch-size 64 \
    --lr 0.0001 \
    --lr-step 4 \
    --lr-gamma 0.9 \
    --in_memory
```

Checkpoints land in `myoutdir/<dataset>/experiment_<n>/`. Run `python train.py --help` for the full set of options (optimizer, scheduler, weight decay, gradient clipping, resuming, …). The same commands are also available as VS Code launch configurations in [`.vscode/launch.json`](.vscode/launch.json).

> **Note:** dataset roots are currently hard-coded in `dataset.py` and `train.py::get_dataloaders`. Point them at your own image directory before training.

## Repository layout

```
├── model.py       EfficientNet-B4 regressor + log-space L1 loss
├── dataset.py     EXIF parsing, RAW/JPEG loading, HDF5 caching, splits
├── train.py       Training loop, W&B logging, checkpointing
├── predict.py     Inference on a folder, EXIF tagging, demo rendering
├── test.py        Evaluation on the held-out split
├── utils.py       Seeding, logging directories, CUDA helpers
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
