# 3rd Party Libraries
import wandb

## Specific Imports
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class Manager:
    """Manager class for fitting the model"""

    def __init__(self, datamodule):
        """Initialize the manager with the given datamodule"""
        self.config = None
        self.datamodule = datamodule

    def __get_trainer(self):
        """Return a trainer with the config from wandb sweep"""
        return Trainer(
            accelerator=self.config.accelerator,
            max_epochs=self.config.max_epochs,
            logger=WandbLogger(),
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

    def __get_model(self, model):
        """Import the model from the models folder"""
        return __import__(f"models.{model}")

    def __init_weights(self):
        """Return a function that initializes the weights of a model"""
        if self.config.weights_init_type == "default":
            # default pytorch init function
            return lambda module: module
        # custom init function
        return lambda module: self.__init_weights_helper(module)

    def __init_weights_helper(self, module):
        """Helper function for __init_weights"""
        init_func = getattr(nn.init, self.config.weights_init_type)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init_func(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        return module

    def run(self):
        """Run the manager using the config from wandb sweep"""
        # setup wandb
        run = wandb.init()
        self.config = run.config

        # setup model with wandb sweep config
        model = self.__get_model(self.config.model)
        model = model(self.config)
        model = model.apply(self.__init_weights)

        # setup trainer
        trainer = self.__get_trainer()

        # fit model
        trainer.fit(model, datamodule=self.datamodule(self.config))

        # finish wandb run
        run.finish()
