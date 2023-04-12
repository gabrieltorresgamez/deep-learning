import torch
import torch.nn as nn
import torch.nn.functional as F

from ._IModel import IModel


class CNN(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        # define dropout
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)
        
        ## 32x32x3 image, 128 filters, 7x7 kernel, 3 padding, output 32x32x128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(128)
        
        ## 32x32x128 image, 64 filters, 7x7 kernel, 3 padding, output 32x32x64
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        ## 32x32x64 image, 32 filters, 7x7 kernel, 3 padding, output 32x32x32
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3_bn = nn.BatchNorm2d(32)
        
        ## 32x32x32, 2x2 kernel, stride 2, output 16x16x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x32 = 8192, output 128 nodes
        self.fc1 = nn.Linear(in_features=(16 * 16 * 32), out_features=512)
        self.fc1_bn = nn.BatchNorm1d(512)
        
        ## 128 nodes input, 64 nodes output
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc2_bn = nn.BatchNorm1d(256)
        
        ## 64 nodes input, 32 nodes output
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc3_bn = nn.BatchNorm1d(128)
        
        ## 32 nodes input, 16 nodes output
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc4_bn = nn.BatchNorm1d(64)
        
        ## 16 nodes input, 10 nodes output
        self.fc5 = nn.Linear(in_features=64, out_features=10)
        self.fc5_bn = nn.BatchNorm1d(10)
        
        # define forward function
        if self.hparams.batch_norm:
            self.forward = self.forward_batchnorm
        else:
            self.forward = self.forward_nobatchnorm

    def forward_nobatchnorm(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x
    
    def forward_batchnorm(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = self.fc5_bn(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x