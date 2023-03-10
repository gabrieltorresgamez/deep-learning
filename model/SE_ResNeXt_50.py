# https://arxiv.org/pdf/1810.00736.pdf TODO: SE-ResNeXt-50
# Implementation Author: https://github.com/maciejbalawejder/Deep-Learning-Collection/blob/main/ConvNets/ResNeXt/resnext_pytorch.py
import torch
import torch.nn as nn

import wandb

from tqdm import tqdm

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.c = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c(x))


# Bottleneck ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinatlity=32):
        super().__init__()
        self.C = cardinatlity
        self.downsample = stride == 2 or first
        res_channels = out_channels // 2
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.C)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)

        self.relu = nn.ReLU()

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.downsample:
            x = self.p(x)

        return self.relu(torch.add(f, x))


# ResNeXt
class ResNeXt(nn.Module):
    def __init__(
        self,
        config_name: int,
        in_channels: int = 3,
        classes: int = 1000,
        C: int = 32,  # cardinality
    ):
        super().__init__()

        configurations = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}

        no_blocks = configurations[config_name]

        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResidualBlock(64, 256, 1, True, cardinatlity=32)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(
                    ResidualBlock(
                        out_features[i - 1], out_features[i], 2, cardinatlity=C
                    )
                )
            for _ in range(no_blocks[i] - 1):
                self.blocks.append(
                    ResidualBlock(out_features[i], out_features[i], 1, cardinatlity=C)
                )

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)


def train(train_data, val_data):
    # Get hyperparameters from wandb
    optimizer = wandb.config.optimizer
    learning_rate = wandb.config.learning_rate
    loss_function = wandb.config.loss_function
    regularization = wandb.config.regularization
    momentum = wandb.config.momentum
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    batch_norm = wandb.config.batch_norm
    device = wandb.config.device
    num_workers = wandb.config.num_workers
    seed = wandb.config.seed

    # Set
    torch.manual_seed(seed)

    # Initialize model
    model = ResNeXt(50, in_channels=3, classes=10, C=32)

    # Move model to device
    model = model.to(device)

    # Initialize optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=regularization,
        )
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=regularization,
            momentum=momentum,
        )
    else:
        raise NotImplementedError

    # Initialize loss function
    if loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # Initialize data loaders
    train_data = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_data = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    len_train = len(val_data.dataset)
    len_val = len(val_data.dataset)

    # Training loop
    for _ in tqdm(range(epochs)):
        # Train model
        model.train()
        train_loss = 0
        train_acc = 0
        for data, target in train_data:
            # Move data to device
            data, target = data.to(device), target.to(device)
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate loss
            loss = loss_function(output, target)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            # Calculate loss
            train_loss += loss.item()
            # Calculate accuracy
            train_acc += torch.sum(pred == target.view_as(pred)).float()

        # Normalize loss and accuracy
        train_loss /= len_train
        train_acc /= len_train

        # Evaluate model
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for data, target in val_data:
                # Move data to device
                data, target = data.to(device), target.to(device)
                # Forward pass
                output = model(data)
                # Calculate loss
                loss = loss_function(output, target)
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                # Calculate loss
                val_loss += loss.item()
                # Calculate accuracy
                val_acc += torch.sum(pred == target.view_as(pred)).float()

            # Normalize loss and accuracy
            val_loss /= len_val
            val_acc /= len_val

        # Log metrics to wandb
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return model
