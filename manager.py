import wandb

import model.MLP as MLP
import model.CNN as CNN
import model.CNN_simple as CNN_simple
import model.SE_ResNeXt_50 as SE_ResNeXt_50

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class Manager:
    def __init__(
        self,
        train_data,
        val_data,
        model,
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.model = model

    def __get_dataloaders(self, batch_size):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=batch_size,
        )
        return train_dataloader, val_dataloader

    def __get_trainer(self, device, epochs):
        return Trainer(
            accelerator=device,
            max_epochs=epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

    def __get_trainer_singlebatch(self, device, epochs):
        return Trainer(
            accelerator=device,
            max_epochs=epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

    def __run_helper(self, model, singlebatch=False):
        run = wandb.init()
        config = run.config
        model = model(config)
        if singlebatch:
            trainer = self.__get_trainer_singlebatch(config.device, config.epochs)
        else:
            trainer = self.__get_trainer(config.device, config.epochs)
        train_dataloader, val_dataloader = self.__get_dataloaders(config.batch_size)
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def run(self):
        if self.model == "CNN_simple":
            self.__run_helper(CNN_simple.CNN)
        elif self.model == "CNN":
            self.__run_helper(CNN.CNN)
        elif self.model == "CNN_singlebatch":
            self.__run_helper(CNN.CNN, singlebatch=True)
        elif self.model == "MLP":
            self.__run_helper(MLP.MLP)
        elif self.model == "SE_ResNeXt_50":
            self.__run_helper(SE_ResNeXt_50.ResNeXt50)
        else:
            raise NotImplementedError
