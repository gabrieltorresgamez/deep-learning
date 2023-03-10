import wandb
import model.MLP as MLP
import model.CNN as CNN
import model.CNN_simple as CNN_simple
import model.CNN_singlebatch as CNN_singlebatch
import model.SE_ResNeXt_50 as SE_ResNeXt_50


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
        # Initialize wandb
        run = wandb.init()
        CNN_simple.train(self.train_data, self.val_data)
        run.finish()

    def __run_CNN(self):
        # Initialize wandb
        run = wandb.init()
        CNN.train(self.train_data, self.val_data)
        run.finish()

    def __run_CNN_singlebatch(self):
        # Initialize wandb
        run = wandb.init()
        CNN_singlebatch.train(self.train_data, self.val_data)
        run.finish()

    def __run_MLP(self):
        # Initialize wandb
        run = wandb.init()
        MLP.train(self.train_data, self.val_data)
        run.finish()

    def __run_SE_ResNeXt_50(self):
        # Initialize wandb
        run = wandb.init()
        SE_ResNeXt_50.train(self.train_data, self.val_data)
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
