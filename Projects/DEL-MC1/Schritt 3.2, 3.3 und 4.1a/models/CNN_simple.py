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
        # 28x28x6 input, output 10
        self.fc1 = nn.Linear(28 * 28 * 6, 10)
        self.fc1_bn = nn.BatchNorm1d(10)

        # define forward function
        if self.hparams.batch_norm:
            self.forward = self.forward_batchnorm
        else:
            self.forward = self.forward_nobatchnorm

    def forward_nobatchnorm(self, x):
        ## output 28x28x6
        x = self.conv1(x)
        x = F.relu(x)
        # flatten
        x = torch.flatten(x, 1)
        ## output 10
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x

    def forward_batchnorm(self, x):
        ## output 28x28x6
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        # flatten
        x = torch.flatten(x, 1)
        ## output 10
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x
