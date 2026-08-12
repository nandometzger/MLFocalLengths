"""Evaluate a checkpoint on the held-out split of the focal length dataset.

Renamed from ``test.py`` so that pytest does not try to collect it as a test
module. Requires the HDF5 dataset built by ``dataset.py``; for predicting on
loose image files use ``predict.py`` instead.

    python evaluate.py --checkpoint myoutdir/My/experiment_0/best_model.pth \
        --hdf5-path data/imgdataset4.h5 --split-file data/split_file4.pickle
"""

import argparse
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import FocalLengthDataset
from model import CNN

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a focal length checkpoint on a held-out split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--data-dir", default="", help="Root image directory (only used to rebuild the cache)")
    parser.add_argument("--hdf5-path", default="data/imgdataset4.h5", help="Cached HDF5 dataset")
    parser.add_argument("--split-file", default="data/split_file4.pickle", help="Train/val/test split")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--in-memory", action="store_true", help="Hold the dataset in RAM")
    return parser


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = self.resolve_device(args.device)

        self.dataloader = self.get_dataloader(args)

        self.model = CNN()
        self.load_checkpoint(args.checkpoint)
        self.model = self.model.to(self.device).eval()

        torch.set_grad_enabled(False)

    @staticmethod
    def resolve_device(choice):
        if choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if choice == "cuda" and not torch.cuda.is_available():
            raise SystemExit("error: --device cuda requested but no CUDA device is available")
        return torch.device(choice)

    @staticmethod
    def get_dataloader(args):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        dataset = FocalLengthDataset(
            root_dir=args.data_dir,
            transform=transform,
            hdf5_path=args.hdf5_path,
            focal_length_path=args.split_file,
            force_recompute=False,
            mode=args.split,
            split_mode="time",
            in_memory=args.in_memory,
        )
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          shuffle=False, drop_last=False)

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        print(f"Checkpoint '{path}' loaded.")

    def evaluate(self):
        stats = defaultdict(float)
        batches = 0

        for sample in tqdm(self.dataloader, desc="Evaluating", unit="batch"):
            sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

            output = self.model(sample)
            _, loss_dict = self.model.get_loss(output[:, 0], sample["y"])

            for key, value in loss_dict.items():
                stats[key] += value
            batches += 1

        return {key: value / max(batches, 1) for key, value in stats.items()}


if __name__ == "__main__":
    args = build_parser().parse_args()

    evaluator = Evaluator(args)

    started = time.time()
    stats = evaluator.evaluate()
    elapsed = time.time() - started

    print(f"Evaluation completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Mean absolute error: {stats.get('lossl1', float('nan')):.2f} mm")
    for key, value in sorted(stats.items()):
        print(f"  {key:20s} {value:.4f}")
