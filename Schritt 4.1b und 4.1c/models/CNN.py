import torch
import torch.nn as nn
import torch.nn.functional as F

from ._IModel import IModel


class CNN(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 28x28x6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        ## 28x28x6, 2x2 kernel, stride 2, output 14x14x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 6x14x14 = 1176, 16 filters, 5x5 kernel, output 10x10x6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        ## 10x10x16 = 1600, 2x2 kernel, stride 2, output 5x5x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 5x5x16 = 400, output 120 nodes
        self.fc1 = nn.Linear(in_features=(5 * 5 * 16), out_features=120)
        
        ## 120 nodes input, 84 nodes output
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        ## 84 nodes input, 10 nodes output
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
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
        
        ## flatten 5x5x16 = 400
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

class CNN2(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 32x32x6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        ## 32x32x6, 2x2 kernel, stride 2, output 16x16x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x6, 16 filters, 5x5 kernel, output 16x16x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)
        ## 16x16x16, 2x2 kernel, stride 2, output 8x8x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8x8x16, output 120 nodes
        self.fc1 = nn.Linear(in_features=(8 * 8 * 16), out_features=120)
        
        ## 120 nodes input, 84 nodes output
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        ## 84 nodes input, 10 nodes output
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
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
        
        ## flatten 5x5x16 = 400
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
    
    
class CNN3(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 32x32x6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        ## 32x32x6, 2x2 kernel, stride 2, output 16x16x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x6, 16 filters, 5x5 kernel, output 16x16x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)
        ## 16x16x16, 2x2 kernel, stride 2, output 8x8x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8x8x16, output 100 nodes
        self.fc1 = nn.Linear(in_features=(8 * 8 * 16), out_features=1000)
        
        ## 120 nodes input, 84 nodes output
        self.fc2 = nn.Linear(in_features=1000, out_features=100)
        
        ## 84 nodes input, 10 nodes output
        self.fc3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
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
        
        ## flatten 5x5x16 = 400
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
    
class CNN4(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 32x32x6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        ## 32x32x6, 2x2 kernel, stride 2, output 16x16x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x6, 16 filters, 5x5 kernel, output 16x16x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        ## 16x16x16, 2x2 kernel, stride 2, output 8x8x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8x8x16, output 100 nodes
        self.fc1 = nn.Linear(in_features=(8 * 8 * 16), out_features=1000)
        
        ## 120 nodes input, 84 nodes output
        self.fc2 = nn.Linear(in_features=1000, out_features=100)
        
        ## 84 nodes input, 10 nodes output
        self.fc3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
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
        
        ## flatten 5x5x16 = 400
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

class CNN5(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 32 filters, 3x3 kernel, 1 padding, output 32x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        ## 32x32x32, 2x2 kernel, stride 2, output 16x16x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x32 image, 16 filters, 3x3 kernel, 1 padding, output 16x16x16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        ## 16x16x16, 2x2 kernel, stride 2, output 8x8x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8x8x16 image, 8 filters, 3x3 kernel, output 8x8x8
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        ## 8x8x8, 2x2 kernel, stride 2, output 4x4x8
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 4x4x8 = 128, output 128 nodes
        self.fc1 = nn.Linear(in_features=(4 * 4 * 8), out_features=128)
        
        ## 128 nodes input, 64 nodes output
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        
        ## 64 nodes input, 32 nodes output
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        
        ## 32 nodes input, 16 nodes output
        self.fc4 = nn.Linear(in_features=32, out_features=16)
        
        ## 16 nodes input, 10 nodes output
        self.fc5 = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x

class CNN6(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 32 filters, 3x3 kernel, 1 padding, output 32x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        ## 32x32x32, 2x2 kernel, stride 2, output 16x16x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 16x16x32 image, 16 filters, 3x3 kernel, 1 padding, output 16x16x16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        ## 16x16x16, 2x2 kernel, stride 2, output 8x8x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8x8x16 image, 8 filters, 3x3 kernel, output 8x8x8
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        ## 4x4x8 = 128, output 128 nodes
        self.fc1 = nn.Linear(in_features=(8 * 8 * 8), out_features=512)
        
        ## 128 nodes input, 64 nodes output
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        
        ## 64 nodes input, 32 nodes output
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        
        ## 32 nodes input, 16 nodes output
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        
        ## 16 nodes input, 10 nodes output
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x

class CNN7(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 128 filters, 3x3 kernel, 1 padding, output 32x32x128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        ## 32x32x128 image, 64 filters, 3x3 kernel, 1 padding, output 32x32x64
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        ## 32x32x64 image, 32 filters, 3x3 kernel, output 32x32x32
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        ## 32x32x32, 2x2 kernel, stride 2, output 16x16x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 4x4x8 = 128, output 128 nodes
        self.fc1 = nn.Linear(in_features=(16 * 16 * 32), out_features=512)
        
        ## 128 nodes input, 64 nodes output
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        
        ## 64 nodes input, 32 nodes output
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        
        ## 32 nodes input, 16 nodes output
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        
        ## 16 nodes input, 10 nodes output
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x

class CNN8(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        
        ## 32x32x3 image, 128 filters, 7x7 kernel, 3 padding, output 32x32x128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        ## 32x32x128 image, 64 filters, 7x7 kernel, 3 padding, output 32x32x64
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        
        ## 32x32x64 image, 32 filters, 7x7 kernel, 3 padding, output 32x32x32
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        ## 32x32x32, 2x2 kernel, stride 2, output 16x16x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 4x4x8 = 128, output 128 nodes
        self.fc1 = nn.Linear(in_features=(16 * 16 * 32), out_features=512)
        
        ## 128 nodes input, 64 nodes output
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        
        ## 64 nodes input, 32 nodes output
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        
        ## 32 nodes input, 16 nodes output
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        
        ## 16 nodes input, 10 nodes output
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
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
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)
        
        ## return output
        return x