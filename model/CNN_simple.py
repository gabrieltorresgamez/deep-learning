import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as li
import torchmetrics as tm


class CNN(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        ## 32x32x3 image, 6 filters, 5x5 kernel, output 28x28x6
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        # 28x28x6 input, output 10
        self.fc1 = nn.Linear(28 * 28 * 6, 10)
        self.fc1_bn = nn.BatchNorm1d(10)

        # set Hyperparameters
        self.hparams.update(hparams)

        # define loss function
        if self.hparams.loss_function == "CrossEntropyLoss":
            self.loss_fn = self.loss_fn_cross_entropy_loss
        else:
            raise NotImplementedError

        # define optimizer
        if self.hparams.optimizer == "Adam":
            self.configure_optimizers = self.configure_optimizers_adam
        elif self.hparams.optimizer == "SGD":
            self.configure_optimizers = self.configure_optimizers_sgd
        else:
            raise NotImplementedError

        # define accuracy
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=10)

        # define f1 score
        self.f1 = tm.F1Score(task="multiclass", num_classes=10, average="macro")

    def forward(self, x):
        ## output 28x28x6
        x = self.conv1(x)
        x = (
            self.conv1_bn(x) if self.hparams.batch_norm else x
        )  # batch norm only if activated
        x = F.relu(x)
        # flatten
        x = torch.flatten(x, 1)
        ## output 10
        x = self.fc1(x)
        x = self.fc1_bn(x) if self.hparams.batch_norm else x
        x = F.log_softmax(x, dim=1)
        ## return output
        return x

    def loss_fn_cross_entropy_loss(self, output, target):
        return nn.CrossEntropyLoss()(output, target)

    def configure_optimizers_adam(self):
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def configure_optimizers_sgd(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.momentum,
        )

    def training_step(self, batch, batch_idx):
        return self.__basic_step(batch, "train_loss", "train_acc", "train_f1")

    def validation_step(self, batch, batch_idx):
        return self.__basic_step(batch, "val_loss", "val_acc", "val_f1")

    def __basic_step(self, batch, name_loss, name_acc, name_f1):
        data, target = batch
        output = self(data)

        loss = self.loss_fn(output, target)
        acc = self.accuracy(output, target)
        f1 = self.f1(output, target)
        self.log(name_loss, loss, on_step=True, on_epoch=True)
        self.log(name_acc, acc, on_step=True, on_epoch=True)
        self.log(name_f1, f1, on_step=True, on_epoch=True)

        return loss
