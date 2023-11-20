import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

# read data
import os
import pandas
import pickle
import numpy as np
import cv2
from PIL import Image

# utils
TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']
CLASS = [2, 7, 12]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ANNOTATION_PATH = 'annotation_split.pkl'
CACHE_PATH = 'cache'

# datasets
class AffectNetDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 img_size: int,
                 type: str,
                 mode: str):

        annotation_path = os.path.join(dataset_dir, ANNOTATION_PATH)
        self.cache_path = os.path.join(dataset_dir, CACHE_PATH)
        os.makedirs(self.cache_path, exist_ok=True)

        annotation = pickle.load(open(annotation_path, 'rb'))
        self.image = annotation[mode]['path']
        self.label = annotation[mode]['label']
        self.face = annotation[mode]['face']
        if type == 'EXPR_Set':
            self.label = self.label.astype(np.int_)
        else:
            self.label = self.label.astype(np.single)

        # preprocess
        if mode == 'Train_Set':
            self.preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, i):
        ## Cache for Cropping
        cache_path = os.path.join(self.cache_path, os.path.basename(self.image[i]))
        if not os.path.exists(cache_path):
            image = Image.open(self.image[i])
            # print(self.image[i])
            image = image.crop(tuple(self.face[i]))
            image.save(cache_path)
        else:
            image = Image.open(cache_path)
        image = self.preprocess(image)
        label = self.label[i]

        return image, label

    def __len__(self):
        return len(self.image)


# datamodules
class AffectNetDataModule(pl.LightningDataModule):

    def __init__(self, params: dict):
        super().__init__()

        self.batch_size = params.get('batch_size', 32)
        self.img_size = params.get('img_size', 224)
        self.data_type = params.get('data_type', 'EXPR_Set')
        self.num_workers = params.get('num_workers', 4)
        self.dataset_dir = params.get('dataset_dir', '/home/user1/dataset/AffectNet')

    def setup(self, stage:str = None) -> None:

        if stage == 'fit':
            self.train_dataset = AffectNetDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Train_Set')

            self.val_dataset = AffectNetDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Validation_Set')

        elif stage == 'validate':
            self.val_dataset = AffectNetDataset(
                self.dataset_dir,
                self.img_size,
                self.data_type,
                'Validation_Set')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
            )


if __name__ == '__main__':
    os.chdir('..')

    dm = AffectNetDataModule({'dataset_dir':'/home/user1/dataset/Aff-Wild/','num_workers':4 , 'data_type':'EXPR_Set'})
    dm.setup('fit')
    dataloader = dm.val_dataloader()
    print(len(dataloader.dataset))
    img, label = next(iter(dataloader))
    print(img.shape, label.shape)
