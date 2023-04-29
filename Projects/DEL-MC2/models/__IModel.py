# 3rd Party Libraries
import torchmetrics as tm
import pytorch_lightning as li

## Specific Imports
from torch import optim
from torch.nn import functional as F


# TODO: Implement
class IEncoder(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # set Hyperparameters
        self.hparams.update(hparams)

        # define metrics
        self.accuracy = tm.Accuracy(task="multiclass")
        self.f1 = tm.F1Score(task="multiclass", average="macro")

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,  # L2 regularization
            momentum=self.hparams.momentum,
        )

    def training_step(self, batch, _):
        return self.__basic_step(batch, "train")

    def validation_step(self, batch, _):
        return self.__basic_step(batch, "val")

    def test_step(self, batch, _):
        return self.__basic_step(batch, "test")

    def __basic_step(self, batch, stage):
        # forward pass
        data, target = batch
        output = self(data)

        # calculate loss and metrics
        loss = self.loss_fn(output, target)
        acc = self.accuracy(output, target)
        f1 = self.f1(output, target)

        # log metrics
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True)
        self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True)

        # return loss
        return loss


# TODO: Implement
class IDecoder(li.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # set Hyperparameters
        self.hparams.update(hparams)

        # define metrics
        self.accuracy = tm.Accuracy(task="multiclass")
        self.f1 = tm.F1Score(task="multiclass", average="macro")

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,  # L2 regularization
            momentum=self.hparams.momentum,
        )

    def training_step(self, batch, _):
        return self.__basic_step(batch, "train")

    def validation_step(self, batch, _):
        return self.__basic_step(batch, "val")

    def test_step(self, batch, _):
        return self.__basic_step(batch, "test")

    def __basic_step(self, batch, stage):
        # forward pass
        data, target = batch
        output = self(data)

        # calculate loss and metrics
        loss = self.loss_fn(output, target)
        acc = self.accuracy(output, target)
        f1 = self.f1(output, target)

        # log metrics
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True)
        self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True)

        # return loss
        return loss
