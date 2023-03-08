import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb


class __CNN(nn.Module):
    def __init__(self, batch_norm=False):
        # Structure of the model
        super().__init__()
        ## 32x32x3 image, 6 filters, 5x5 kernel, stride 1, padding 0, output 28x28x6
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.conv1_bn = nn.BatchNorm2d(6)
        ## 6x28x28, 2x2 kernel, stride 2, padding 0, output 6x14x14
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1_bn = nn.BatchNorm2d(6)
        ## 6x14x14 = 1176, 100 nodes output
        self.fc1 = nn.Linear(6 * 14 * 14, 100)
        self.fc1_bn = nn.BatchNorm1d(100)
        ## 100 nodes input, 10 nodes output
        self.fc2 = nn.Linear(100, 10)

        # config batch normalization
        self.batch_norm = batch_norm

    def forward(self, x):
        # Forward pass
        ## output 28x28x32
        x = self.conv1(x)
        x = self.conv1_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 14x14x32 = 6272
        x = self.pool1(x)
        x = self.pool1_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = torch.flatten(x, 1)
        ## output 100
        x = self.fc1(x)
        x = self.fc1_bn(x) if self.batch_norm else x  # batch norm only if specified
        x = F.relu(x)
        ## output 10 classes
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
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
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    batch_norm = wandb.config.batch_norm
    device = wandb.config.device
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
            model.parameters(), lr=learning_rate, weight_decay=regularization
        )
    elif optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=regularization
        )
    else:
        raise NotImplementedError

    # Set loss function
    if loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # set batch size using dataloader
    train_data = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
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
        train_loss /= len(train_data.dataset)
        train_acc /= len(train_data.dataset)

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
            val_loss /= len(val_data.dataset)
            val_acc /= len(val_data.dataset)

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
