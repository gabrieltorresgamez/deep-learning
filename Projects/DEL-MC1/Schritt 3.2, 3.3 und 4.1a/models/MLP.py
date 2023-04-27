import torch
import torch.nn as nn
import torch.nn.functional as F

from ._IModel import IModel


class MLP(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        ## 32x32x3 image = 2048, output 5000
        self.fc1 = nn.Linear(32 * 32 * 3, 5000)
        self.fc1_bn = nn.BatchNorm1d(5000)
        # input 5000, output 5000
        self.fc2 = nn.Linear(5000, 5000)
        self.fc2_bn = nn.BatchNorm1d(5000)
        # input 5000, output 1000
        self.fc3 = nn.Linear(5000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)
        # input 1000, output 100
        self.fc4 = nn.Linear(1000, 100)
        self.fc4_bn = nn.BatchNorm1d(100)
        # input 100, output 10
        self.fc5 = nn.Linear(100, 10)
        self.fc5_bn = nn.BatchNorm1d(10)

        # define forward function
        if self.hparams.batch_norm:
            self.forward = self.forward_batchnorm
        else:
            self.forward = self.forward_nobatchnorm

    def forward_nobatchnorm(self, x):
        # flatten input
        x = torch.flatten(x, 1)
        # input 2048, output 5000
        x = self.fc1(x)
        x = F.relu(x)
        # input 5000, output 5000
        x = self.fc2(x)
        x = F.relu(x)
        # input 5000, output 1000
        x = self.fc3(x)
        x = F.relu(x)
        # input 1000, output 100
        x = self.fc4(x)
        x = F.relu(x)
        # input 100, output 10
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x

    def forward_batchnorm(self, x):
        # flatten input
        x = torch.flatten(x, 1)
        # input 2048, output 5000
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        # input 5000, output 5000
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        # input 5000, output 1000
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        # input 1000, output 100
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(x)
        # input 100, output 10
        x = self.fc5(x)
        x = self.fc5_bn(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x
