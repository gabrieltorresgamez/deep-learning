import wandb
import model.CNN as CNN
import model.CNN_singlebatch as CNN_singlebatch


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

    def __run_RNN(self):
        raise NotImplementedError

    def run(self):
        if self.model == "CNN":
            self.__run_CNN()
        elif self.model == "CNN_singlebatch":
            self.__run_CNN_singlebatch()
        elif self.model == "RNN":
            self.__run_RNN()
        else:
            raise NotImplementedError
