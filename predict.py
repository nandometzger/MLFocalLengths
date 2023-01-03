import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize  
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
   
from dataset import ImageFolder

from utils import to_cuda
from model import CNN

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate') 
parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes') 

class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cuda = torch.cuda.is_available()

        self.dataloader = self.get_dataloader(args)

        self.model = CNN()
        self.model = self.model.to("cuda") if self.cuda else self.model

        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)

        for sample in self.dataloader:
            sample = to_cuda(sample)

            output = self.model(sample)

            print(sample["path"][0][0] ,"Predicted", output.cpu().item(), "mm")

        return None

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        
        dataset = ImageFolder(args.root_dir)

        return DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = parser.parse_args()
    print(parser.format_values())
    
    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print(stats)