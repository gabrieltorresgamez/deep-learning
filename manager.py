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

    def __run_CNN_simple(self):
        run = wandb.init()
        config = run.config
        model = CNN_simple.CNN(config)
        trainer = Trainer(
            accelerator=config.device,
            max_epochs=config.epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=config.batch_size,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def __run_CNN(self):
        run = wandb.init()
        config = run.config
        model = CNN.CNN(config)
        trainer = Trainer(
            accelerator=config.device,
            max_epochs=config.epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=config.batch_size,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def __run_CNN_singlebatch(self):
        run = wandb.init()
        config = run.config
        model = CNN.CNN(config)
        trainer = Trainer(
            accelerator=config.device,
            max_epochs=config.epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=config.batch_size,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def __run_MLP(self):
        run = wandb.init()
        config = run.config
        model = MLP.MLP(config)
        trainer = Trainer(
            accelerator=config.device,
            max_epochs=config.epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=config.batch_size,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def __run_SE_ResNeXt_50(self):
        run = wandb.init()
        config = run.config
        model = SE_ResNeXt_50.ResNeXt50(config)
        trainer = Trainer(
            accelerator=config.device,
            max_epochs=config.epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
        )
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=config.batch_size,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        run.finish()

    def run(self):
        if self.model == "CNN_simple":
            self.__run_CNN_simple()
        elif self.model == "CNN":
            self.__run_CNN()
        elif self.model == "CNN_singlebatch":
            self.__run_CNN_singlebatch()
        elif self.model == "MLP":
            self.__run_MLP()
        elif self.model == "SE_ResNeXt_50":
            self.__run_SE_ResNeXt_50()
        else:
            raise NotImplementedError
