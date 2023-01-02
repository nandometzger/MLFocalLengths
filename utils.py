

import random
import numpy as np
import torch

import os

import csv

from pylab import figure, imshow, matshow, grid, savefig, colorbar

from torchvision.transforms import Normalize
denorm = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Middlebury


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])

def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder


def to_cuda(sample):
    sampleout = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sampleout[key] = val.cuda()
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.cuda())
                else:
                    new_val.append(val)
            sampleout[key] = new_val
        else:
            sampleout[key] = val
    return sampleout


def plot_2dmatrix(matrix, fig=1, vmin=None, vmax=None):
    if torch.is_tensor(matrix):
        if matrix.is_cuda:
            matrix = matrix.cpu()
        matrix = matrix.numpy()
    figure(fig)
    matshow(matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
    grid(True)
    colorbar()
    savefig('plot_outputs/last_plot.png')
