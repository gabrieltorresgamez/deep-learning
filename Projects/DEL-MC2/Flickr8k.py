## Specific Imports
from PIL import Image

## Specific Imports
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule, LightningDataModule


class Flickr8k(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.captions = []

        with open(ann_file, "r") as f:
            for line in f:
                image, caption = line.strip().split(";")
                self.images.append(image)
                self.captions.append(caption)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        image = Image.open(f"{self.root}/{filename}").convert("RGB")
        caption = self.captions[idx]
        if self.transform:
            image = self.transform(image)

        return filename, image, caption


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
