import torch
import torch.nn as nn
import torch.nn.functional as F

from ._IModel import IModel


class CNN(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 28x28x6
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        ## 6x28x28, 2x2 kernel, stride 2, padding 0, output 6x14x14
        self.pool1 = nn.MaxPool2d(2, 2)
        ## 6x14x14 = 1176, 16 filters, 5x5 kernel, output 6x10x10
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        ## 16x10x10 = 1600, 2x2 kernel, stride 2, padding 0, output 16x5x5
        self.pool2 = nn.MaxPool2d(2, 2)
        ## 16x5x5 = 400, output 120 nodes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        ## 120 nodes input, 84 nodes output
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        ## 84 nodes input, 10 nodes output
        self.fc3 = nn.Linear(84, 10)

        # define forward function
        if self.hparams.batch_norm:
            self.forward = self.forward_batchnorm
        else:
            self.forward = self.forward_nobatchnorm

    def forward_nobatchnorm(self, x):
        ## output 28x28x6
        x = self.conv1(x)
        x = F.relu(x)
        ## output 14x14x6
        x = self.pool1(x)
        ## output 10x10x16
        x = self.conv2(x)
        x = F.relu(x)
        ## output 5x5x16
        x = self.pool2(x)
        ## flatten 10x10x6 = 600
        x = torch.flatten(x, 1)
        ## output 120
        x = self.fc1(x)
        x = F.relu(x)
        ## output 84
        x = self.fc2(x)
        x = F.relu(x)
        ## output 10
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x

    def forward_batchnorm(self, x):
        ## output 28x28x6
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        ## output 14x14x6
        x = self.pool1(x)
        ## output 10x10x16
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        ## output 5x5x16
        x = self.pool2(x)
        ## flatten 10x10x6 = 600
        x = torch.flatten(x, 1)
        ## output 120
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        ## output 84
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        ## output 10
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x
