import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from einops import rearrange
import numpy as np
import os
import pandas as pd

class KannadaMnistDataSet(Dataset):
    def __init__(self, set, transform, root='data'):

        self.data_dir = os.path.join(root, set)
        self.data = pd.read_csv(
            os.path.join(self.data_dir, os.listdir(self.data_dir)[0])
            )
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:]
        image = np.array(data, np.float32)
        image = data/255.
        iamge = rearrange(data, '(i h w) -> i h w', i=1, h=28)
        label = self.data.iloc[idx, 0]
        return {
            'label': np.array(label, dtype=np.int64),
            'features': data
        }

def build_dataloaders(batch_size):
    train_dataset = KannadaMnistDataSet('train')
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
    test_dataset = KannadaMnistDataSet('test')
    hard_test_dataset = KannadaMnistDataSet('hard_test')

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

    train_dataset = KannadaMnistDataSet('train')

    train, val = torch.utils.data.random_split(train_dataset, [55000, 5000])
    print(len(train))
    print(len(val))

    train_dataloader = DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    # print(len(train_dataset))
    n_samples = 0
    for i, data in enumerate(train_dataloader):
        labels, features = data['label'], data['features']
        n_samples += 1
        if i == 0:
            print(features)
            print(labels)
            print(f"Labels batch shape: {labels.size()}")
            print(f"Features batch shape: {features.size()}")

    print(n_samples)