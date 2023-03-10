import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb

from model.CNN import __CNN


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

    # Iterate over only one batch
    data, target = next(iter(train_data))
    # Move data to device
    data, target = data.to(device), target.to(device)

    # Iterate over only one batch
    data_val, target_val = next(iter(val_data))
    # Move data to device
    data_val, target_val = data_val.to(device), target_val.to(device)

    # Iterate over epochs
    for _ in range(epochs):
        # Train model
        model.train()
        train_loss = 0
        train_acc = 0

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
        train_loss = loss.item() / batch_size
        # Calculate accuracy
        train_acc = torch.sum(pred == target.view_as(pred)).float() / batch_size

        # Evaluate model
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            # Forward pass
            output = model(data_val)
            # Calculate loss
            loss = loss_function(output, target_val)
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            # Calculate loss
            val_loss = loss.item() / batch_size
            # Calculate accuracy
            val_acc = torch.sum(pred == target_val.view_as(pred)).float() / batch_size

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
