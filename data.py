import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
import numpy as np
import os
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

class KannadaMnistDataSet(Dataset):
    def __init__(self, dataset, transform=None, root='data'):

        self.data_dir = os.path.join(root, dataset)
        self.data = pd.read_csv(
            os.path.join(self.data_dir, os.listdir(self.data_dir)[0])
            )
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:]
        image = np.array(image, np.uint8)
        image = rearrange(image, '(h w i) -> h w i', i=1, h=28)
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx, 0]
        return np.array(label, dtype=np.int64), image

def plot_images(dataset, nrows, ncols, transform, set='train'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20))
    for i in range(nrows):
        for j in range(ncols):
            label, image = dataset[j+ncols*i]
            image = rearrange(image, '() h w -> h w')
            ax[i][j].axis('off')
            ax[i][j].imshow(image, cmap='gray')
    plt.savefig(f'images/{set}_set_first_100_samples' + transform * '_transformed'+ '.png')

def plot_image(image, idx):
    image = rearrange(image, '() h w -> h w')
    plt.imshow(image, cmap='gray')
    plt.savefig(f'indiv_img_{idx}.png')

def build_transforms(data_mean, data_std):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
        transforms.RandomAffine(
            degrees=10,
            scale=(0.9, 1.1), 
            shear=5)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std),
    ])

    return train_transform, test_transform

def build_dataloaders(batch_size):
    train_dataset = KannadaMnistDataSet('train')
    
    df = train_dataset.data/255.
    df.drop(columns='label', inplace=True)
    array = df.to_numpy()
    data_mean = array.mean()
    data_std = array.std()

    train_transform, test_transform = build_transforms(data_mean, data_std)

    train_dataset = KannadaMnistDataSet('train', transform=train_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
    test_dataset = KannadaMnistDataSet('test', transform=test_transform)
    hard_test_dataset = KannadaMnistDataSet('hard_test', transform=test_transform)

    return {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
            ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=0
            ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            num_workers=0
            ),
        'hard_test': DataLoader(
            hard_test_dataset, 
            batch_size=batch_size, 
            num_workers=0
            )
    }

if __name__ == '__main__':

    transform=False
    train_dataset = KannadaMnistDataSet('train')

    plot_images(train_dataset, 10, 10, transform, 'train')