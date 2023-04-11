import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as li
import torchmetrics as tm


class IModel(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # set Hyperparameters
        self.hparams.update(hparams)

        # define loss function
        if self.hparams.loss_function == "CrossEntropyLoss":
            self.loss_fn = self.__loss_fn_cross_entropy_loss
        else:
            raise NotImplementedError

        # define optimizer
        if self.hparams.optimizer == "Adam":
            self.configure_optimizers = self.__configure_optimizers_adam
        elif self.hparams.optimizer == "SGD":
            self.configure_optimizers = self.__configure_optimizers_sgd
        else:
            raise NotImplementedError

        # define accuracy
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=10)

        # define f1 score
        self.f1 = tm.F1Score(task="multiclass", num_classes=10, average="macro")

    def __loss_fn_cross_entropy_loss(self, output, target):
        return nn.CrossEntropyLoss()(output, target)

    def __configure_optimizers_adam(self):
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,  # L2 regularization
        )

    def __configure_optimizers_sgd(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,  # L2 regularization
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
