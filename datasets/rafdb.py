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
from PIL import Image

# utils
TYPE = ['VA_Set', 'EXPR_Set', 'AU_Set']
CLASS = [2, 7, 12]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ANNOTATION_PATH = 'list_patition_label.txt'

# datasets
class RAFDBDataset(Dataset):
    def __init__(self,
                 img_size: int,
                 image: dict,
                 label: dict,
                 mode: str):

        self.image = image[mode]
        self.label = label[mode]

        self.label = np.array(self.label).astype(np.int_)

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
        image = Image.open(self.image[i])
        image = self.preprocess(image)
        label = self.label[i]

        return image, label

    def __len__(self):
        return len(self.image)


# datamodules
class RAFDBDataModule(pl.LightningDataModule):

    def __init__(self, params: dict):
        super().__init__()

        self.batch_size = params.get('batch_size', 32)
        self.img_size = params.get('img_size', 224)
        self.data_type = params.get('data_type', 'EXPR_Set')
        self.num_workers = params.get('num_workers', 4)
        self.dataset_dir = params.get('dataset_dir', '/home/user1/dataset/RAF-DB/')

        self.image = {'Train_Set':[],'Validation_Set':[]}
        self.label = {'Train_Set':[],'Validation_Set':[]}

        annotation_path = os.path.join(self.dataset_dir, ANNOTATION_PATH)
        with open(annotation_path) as f:
            for line in f.readlines():
                item = line.split()
                path = os.path.join(self.dataset_dir,'aligned',item[0][:-4]+'_aligned.jpg')
                mode = 'Train_Set' if 'train' in item[0] else 'Validation_Set'
                self.image[mode].append(path)
                self.label[mode].append(int(item[1]) - 1)


    def setup(self, stage:str = None) -> None:

        if stage == 'fit':
            self.train_dataset = RAFDBDataset(
                self.img_size,
                self.image,
                self.label,
                'Train_Set')

            self.val_dataset = RAFDBDataset(
                self.img_size,
                self.image,
                self.label,
                'Validation_Set')

        elif stage == 'validate':
            self.val_dataset = RAFDBDataset(
                self.img_size,
                self.image,
                self.label,
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
            pin_memory=True)


if __name__ == '__main__':
    os.chdir('..')

    dm = RAFDBDataModule({'num_workers':4 , 'data_type':'EXPR_Set'})
    dm.setup('fit')
    dataloader = dm.val_dataloader()
    print(len(dataloader.dataset))
    img, label = next(iter(dataloader))
    print(img.shape, label.shape)
