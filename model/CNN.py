import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb

# TODO: Add 1 convolutional layer
class __CNN(nn.Module):
    def __init__(self, batch_norm=False):
        # Structure of the model
        super().__init__()
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

        # config batch normalization
        self.batch_norm = batch_norm

    def forward(self, x):
        # Forward pass
        ## output 28x28x6
        x = self.conv1(x)
        x = self.conv1_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 14x14x6
        x = self.pool1(x)
        ## output 10x10x16
        x = self.conv2(x)
        x = self.conv2_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 5x5x16
        x = self.pool2(x)
        ## flatten 10x10x6 = 600
        x = torch.flatten(x, 1)
        ## output 120
        x = self.fc1(x)
        x = self.fc1_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 84
        x = self.fc2(x)
        x = self.fc2_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 10
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        ## return output
        return x


def train(
    train_data,
    val_data,
):
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

    # Set model
    model = __CNN(batch_norm).to(device)

    # Set seed
    torch.manual_seed(seed)

    # Set device
    device = torch.device(device)

    # Set optimizer
    if optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=regularization,
        )
    elif optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=regularization,
            momentum=momentum,
        )
    else:
        raise NotImplementedError

    # Set loss function
    if loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    len_train = len(train_data)
    len_val = len(val_data)

    # set batch size using dataloader
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

    # Iterate over epochs
    for _ in range(epochs):
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
