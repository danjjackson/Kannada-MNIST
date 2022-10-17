import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchdata.datapipes as dp
from einops import rearrange
import numpy as np
import os
import pandas as pd

BATCH_SIZE = 8

def row_processor(row):
    data = np.array(row[1:], np.float32)
    data = data/255.
    data = rearrange(data, '(i h w) -> i h w', i=1, h=28)
    return {
        'label': np.array(row[0], dtype=np.int64),
        'data': data
    }

def build_datapipes(set, root_dir='data'):
    data_dir = os.path.join(root_dir, set)
    data_pipe = dp.iter.FileLister(data_dir)
    data_pipe = data_pipe.open_files(mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()
    data_pipe = data_pipe.map(row_processor)
    return data_pipe

def build_dataloaders():
    dataloaders = {}
    for set in ['train', 'val', 'test']:
        dataloaders[set] = DataLoader(
            dataset=build_datapipes(set),
            batch_size=BATCH_SIZE,
        )
    return dataloaders


class KannadaMnistDataSet(VisionDataset):
    def __init__(self, root, mode):

        self.data_dir = os.path.join(root, mode)
        self.data = pd.read_csv(
            os.path.join(self.data_dir, os.listdir(self.data_dir)[0])
            )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, 1:]
        data = np.array(data, np.float32)
        data = data/255.
        data = rearrange(data, '(i h w) -> i h w', i=1, h=28)
        label = self.data.iloc[idx, 0]
        return {
            'label': np.array(label, dtype=np.int64),
            'data': data
        }

if __name__ == '__main__':

    train_dataset = KannadaMnistDataSet('data', 'train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(len(train_dataset))
    for i, data in enumerate(train_dataloader):
        labels, features = data['label'], data['data']
        if i == 0:
            print(features)
            print(labels)
            print(f"Labels batch shape: {labels.size()}")
            print(f"Features batch shape: {features.size()}")




    # dataloaders = build_dataloaders()

    # first = next(iter(dataloaders['val']))
    # labels, features = first['label'], first['data']
    # print(features)
    # print(labels)
    # print(f"Labels batch shape: {labels.size()}")
    # print(f"Features batch shape: {features.size()}")

    # n_sample = 0
    # for row in iter(dataloaders['test']):
    #     n_sample += 1
    # print(f'n_sample = {n_sample}')

