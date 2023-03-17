import wandb

import utils.WeightsInit as WeightsInit

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
    ):
        self.train_data = train_data
        self.val_data = val_data

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
            precision=16,
            accelerator=device,
            max_epochs=epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

    def __get_trainer_singlebatch(self, device, epochs):
        return Trainer(
            precision=16,
            accelerator=device,
            max_epochs=epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

    def __get_model(self, model):
        if model in ["CNN", "CNN_singlebatch"]:
            return CNN.CNN
        elif model == "CNN_simple":
            return CNN_simple.CNN
        elif model == "MLP":
            return MLP.MLP
        elif model == "SE_ResNeXt_50":
            return SE_ResNeXt_50.ResNeXt50
        else:
            raise NotImplementedError

    def __init_weights_func(self, weights_init_type):
        return WeightsInit.WeightsInit(weights_init_type).init_weights

    def run(self):
        # setup wandb
        run = wandb.init()
        config = run.config

        # setup model with wandb sweep config
        model = self.__get_model(config.model)
        model = model(config)
        model = model.apply(self.__init_weights_func(config.weights_init_type))
        # model = torch.compile(model)  # PyTorch 2.0 awaiting fix https://github.com/pytorch/pytorch/pull/96980

        # setup trainer
        if "singlebatch" in config.model:
            trainer = self.__get_trainer_singlebatch(config.device, config.epochs)
        else:
            trainer = self.__get_trainer(config.device, config.epochs)
        # setup dataloaders
        train_dataloader, val_dataloader = self.__get_dataloaders(config.batch_size)
        # train model using pytorch lightning
        trainer.fit(model, train_dataloader, val_dataloader)
        # finish wandb run
        run.finish()
