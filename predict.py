import os
import argparse
from collections import defaultdict
import time

import torch
from torchvision.transforms import Normalize  
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import piexif
from PIL import Image
from pylab import figure, imshow, matshow, grid, savefig, colorbar, subplot, title


from dataset import ImageFolder

from utils import to_cuda
from model import CNN

import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate') 
parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
parser.add_argument('--save_demo', type=bool, default=False, help='Root directory of the dataset')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes') 

class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cuda = torch.cuda.is_available()

        self.dataloader = self.get_dataloader(args)

        self.model = CNN()

        self.resume(path=args.checkpoint)
        self.model = self.model.to("cuda") if self.cuda else self.model
        self.model.eval()

        torch.set_grad_enabled(False)

    def evaluate(self):
        test_stats = defaultdict(float)

        for sample in self.dataloader:
            sample = to_cuda(sample) if self.cuda else sample

            output = self.model(sample)
            
            if sample["path"][0][0].split(".")[-1] in ["jpg", "JPG"]:
                exif_dict = piexif.load(sample["path"][0][0])
                exif_dict["MLFocalLength"] = int(output.cpu().numpy().item()+0.5) 
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, sample["path"][0][0])

            else:
                print("metadata tagging only supported for jpgs")

            if self.args.save_demo:
                subplot(1,1,1)
                imshow(sample["raw_img"][0].cpu().numpy())
                title("{:2.1f}mm".format(output.cpu().item()))
                folder_name = "/".join(sample["path"][0][0].split("/")[:-2]+[sample["path"][0][0].split("/")[-2]+"_taged"])
                filename = "taged_" + sample["path"][0][0].split("/")[-1]
                outfile = os.path.join(folder_name,filename)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name) 
                savefig(outfile)

            print(sample["path"][0][0] , "Predicted {:5.1f}mm".format(output.cpu().item()))

        return None

    @staticmethod
    def get_dataloader(args: argparse.Namespace):
        
        dataset = ImageFolder(args.root_dir)

        return DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False)

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path, map_location=torch.device('cuda') if self.cuda else torch.device('cpu'))
        # self.load_state_dict(torch.load(PATH, map_location=device))
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