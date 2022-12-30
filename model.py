

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3) 
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.conv5 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.lossl1 = torch.nn.L1Loss()
        self.lossl2 = torch.nn.MSELoss()
        self.log_transform = True
        self.soft_plus = nn.Softplus()
        

    def forward(self, x):
        x = x["img"]
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 16 * 6 * 6)
        x = x.mean((2,3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_loss(self, pred, gt, eps=1e-7): 
        mean_pred = pred.mean()

        # transform to log in case of log loss 
        if self.log_transform:
            pred = self.soft_plus(pred)
            pred_log = torch.log(pred+eps)
            gt_log = torch.log(gt+eps)
            lossl1_log = self.lossl1(pred_log,gt_log)
            lossl2_log = self.lossl2(pred_log,gt_log)

        lossl1 = self.lossl1(pred,gt)
        lossl2 = self.lossl2(pred,gt)

        optimization_loss = lossl1

        return optimization_loss, {
            "lossl1": lossl1,  "lossl2": lossl2, 
            "lossl1_log": lossl1_log,  "lossl2_log": lossl2_log, 
            "mean_pred": mean_pred, 
            "optimization_loss": optimization_loss}



# Define a simple convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
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

        # transform to log in case of log loss 
        if self.log_transform:
            pred = self.soft_plus(pred)
            pred_log = torch.log(pred+eps)
            gt_log = torch.log(gt+eps)
            lossl1_log = self.lossl1(pred_log,gt_log)
            lossl2_log = self.lossl2(pred_log,gt_log)

        lossl1 = self.lossl1(pred,gt)
        lossl2 = self.lossl2(pred,gt)

        optimization_loss = lossl1

        return optimization_loss, {
            "lossl1": lossl1,  "lossl2": lossl2, 
            "lossl1_log": lossl1_log,  "lossl2_log": lossl2_log, 
            "mean_pred": mean_pred, 
            "optimization_loss": optimization_loss}