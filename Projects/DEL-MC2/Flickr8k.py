## Standard Libraries
from PIL import Image

## 3rd Party Libraries
import torch

## Specific Imports
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningDataModule, LightningDataModule
from tqdm import tqdm
from nltk.tokenize import word_tokenize


class Flickr8k(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.transform = transform
        self.dict = {}
        self.images = {}

        # read the annotations file
        ann = open(ann_file, "r", encoding="utf-8").read().split("\n")
        for line in ann:
            if len(line):
                file, caption = line.split(";")
                if file not in self.dict:
                    self.dict[file] = [word_tokenize(caption.lower())]
                else:
                    self.dict[file].append(word_tokenize(caption.lower()))

        # preload the images
        for file in tqdm(self.dict):
            image = Image.open(f"{self.root}/{file}").convert("RGB")
            if transform:
                image = transform(image)
            self.images[file] = image

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        file, captions = list(self.dict.items())[idx]
        image = self.images[file]
        return file, image, captions


class DataModuleFlickr8k(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        preload = self.config is not None

        self.data = Flickr8k(
            root="data/images",
            ann_file="data/captions.txt",
            transform=transform,
        )

        return self

    def setup(self):
        generator = Generator().manual_seed(self.config.seed)
        self.train_data, self.val_data, self.test_data = random_split(
            self.data, [0.7, 0.1, 0.2], generator=generator
        )

    def train_dataloader(self):
        return self.__get_dataloader(self.train_data)

    def val_dataloader(self):
        return self.__get_fullbatch_dataloader(self.val_data)

    def test_dataloader(self):
        return self.__get_fullbatch_dataloader(self.test_data)

    def __get_dataloader(self, data):
        return DataLoader(
            data,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def __get_fullbatch_dataloader(self, data):
        return DataLoader(
            data,
            batch_size=len(data),
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )
