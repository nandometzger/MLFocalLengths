

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights


# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # self.model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(1000, 1)
        
        self.lossl1 = torch.nn.L1Loss()
        self.lossl2 = torch.nn.MSELoss()
        self.log_transform = True
        self.soft_plus = nn.Softplus()

    def forward(self, x):
        x = x["img"]
        x = self.model(x)
        x = self.fc1(x)
        return x

    def get_loss(self, pred, gt, eps=1e-7): 
        mean_pred = pred.mean()
        mean_gt = gt.to(float).mean()

        lossl1 = self.lossl1(pred,gt)
        lossl2 = self.lossl2(pred,gt)

        # transform to log in case of log loss 
        if self.log_transform:
            pred = self.soft_plus(pred)
            pred_log = torch.log(pred+eps)
            gt_log = torch.log(gt+eps)
            lossl1_log = self.lossl1(pred_log,gt_log)
            lossl2_log = self.lossl2(pred_log,gt_log)

        optimization_loss = lossl1_log

        return optimization_loss, {
            "lossl1": lossl1.detach().item(),  "lossl2": lossl2.detach().item(), 
            "lossl1_log": lossl1_log.detach().item(),  "lossl2_log": lossl2_log.detach().item(), 
            "mean_pred": mean_pred.detach().item(), "mean_gt": mean_gt.detach().item(), 
            "optimization_loss": optimization_loss.detach().item()}
