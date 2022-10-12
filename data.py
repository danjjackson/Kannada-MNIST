import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from einops import rearrange
import numpy as np
import os

BATCH_SIZE = 1

def row_processor(row):
    data = np.array(row[1:], np.float64)
    data = rearrange(data, '(h w) -> h w', h=28)
    return {
        'label': np.array(row[0], np.int32),
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
            num_workers=1
        )
    return dataloaders

if __name__ == '__main__':

    dataloaders = build_dataloaders()

    # first = next(iter(dataloaders['val']))
    # labels, features = first['label'], first['data']
    # print(f"Labels batch shape: {labels.size()}")
    # print(f"Features batch shape: {features.size()}")

    n_sample = 0
    for row in iter(dataloaders['test']):
        n_sample += 1
    print(f'n_sample = {n_sample}')