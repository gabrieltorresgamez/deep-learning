import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as li
import torchmetrics as tm


class MLP(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()
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
        self.fc4_bn = nn.BatchNorm1d(10)

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
        self.f1 = tm.F1Score(task="multiclass", num_classes=10)

    def forward(self, x):
        # flatten input
        x = torch.flatten(x, 1)
        # input 2048, output 5000
        x = self.fc1(x)
        x = self.fc1_bn(x) if self.hparams.batch_norm else x
        x = F.relu(x)
        # input 5000, output 5000
        x = self.fc2(x)
        x = self.fc2_bn(x) if self.hparams.batch_norm else x
        x = F.relu(x)
        # input 5000, output 1000
        x = self.fc3(x)
        x = self.fc3_bn(x) if self.hparams.batch_norm else x
        x = F.relu(x)
        # input 1000, output 100
        x = self.fc4(x)
        x = self.fc4_bn(x) if self.hparams.batch_norm else x
        x = F.relu(x)
        # input 100, output 10
        x = self.fc5(x)
        x = self.fc5_bn(x) if self.hparams.batch_norm else x
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
        self.log(name_loss, loss)
        self.log(name_acc, acc)
        self.log(name_f1, f1)

        return loss
