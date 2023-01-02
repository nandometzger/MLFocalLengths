import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize  
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
   
from dataset import FocalLengthDataset

from utils import to_cuda
from model import CNN

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--data-dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--in_memory', default=False, action='store_true', help='Hold data in memory during evaluation')

class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloader = self.get_dataloader(args)

        self.model = CNN()
        self.model = self.model.to("cuda") if self.cuda else self.model

        self.resume(path=args.checkpoint)
        self.model.cuda().eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)

        for sample in tqdm(self.dataloader, leave=False):
            sample = to_cuda(sample)

            output = self.model(sample)

            _, loss_dict = self.model.get_loss(output, sample)

            for key in loss_dict:
                test_stats[key] += loss_dict[key]

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
    
        if args.dataset == 'My':
            data_transform_eval = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
            ])

            dataset = FocalLengthDataset(root_dir=r'C:\Users\nando\Pictures\SD Kartenbackups\All_hierarchical\2022',
                transform=data_transform_eval, hdf5_path="data/imgdataset3.h5", focal_length_path='data/split_file3.pickle',
                force_recompute=False, mode="test", split_mode="time", in_memory=args.in_memory)
        elif args.dataset == "folder":
            dataset = ImageFolder()
            
        else:
            raise NotImplementedError(f'Dataset {args.dataset}')

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

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
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)