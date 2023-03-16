# https://lightning.ai/docs/pytorch/latest/advanced/transfer_learning.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as li
import torchmetrics as tm

import torchvision.models as models


class ResNeXt50(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnext50_32x4d(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

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
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
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
