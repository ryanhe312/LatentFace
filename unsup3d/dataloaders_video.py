import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 16)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 128)
    crop = cfgs.get('crop', None)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', '/home/user1/dataset/unzippedFaces/')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', '/home/user1/dataset/unzippedFaces/test')

    train_loader = val_loader = test_loader = None
    get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        val_data_dir = os.path.join(train_val_data_dir, "val")
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_loader(data_dir=train_data_dir, num_workers=num_workers, is_validation=False)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_loader(data_dir=val_data_dir, num_workers=num_workers, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS) and ('recrop' in filename)


## simple image dataset ##
def make_dataset(dir, batchsize=16):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = {}
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                if root in images:
                    images[root].append(fname)
                else:
                    images[root]=[fname]

    for key in list(images.keys()):
        if len(images[key])<batchsize:
            del images[key]
    return images

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 image_size=128,
                 crop=None,
                 is_validation=False,
                 batch_size=16):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.paths = make_dataset(data_dir, batch_size)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation
        self.batch_size = batch_size

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        fnames = list(self.paths.values())[index % self.size]
        froot = list(self.paths.keys())[index % self.size]

        select_names = np.random.choice(fnames,self.batch_size)
        # select_names = fnames
        select_imgs = []
        for name in select_names:
            fpath = os.path.join(froot,name)
            img = Image.open(fpath).convert('RGB')
            hflip = not self.is_validation and np.random.rand()>0.5
            select_imgs.append(self.transform(img, hflip=hflip))

        return torch.stack(select_imgs)

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'


def get_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None):

    dataset = ImageDataset(
        data_dir,
        image_size=image_size,
        crop=crop,
        is_validation=is_validation,
        batch_size = batch_size
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory= True,
        shuffle=not is_validation,
        num_workers=num_workers,
        collate_fn=lambda x:x[0]
    )
    return loader

if __name__ == '__main__':
    train, val, test = get_data_loaders({'run_train':True,'run_test':True})
    print(len(train),len(val),len(test))
    print(next(iter(train)).shape)

