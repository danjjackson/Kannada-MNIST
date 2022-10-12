import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp

import numpy as np
import os

def row_processor(row):
    return {
        'label': np.array(row[0], np.int32),
        'data': np.array(row[1:], np.float64)
    }

def build_datapipes(root_dir='data', set='train'):
    data_dir = os.path.join(root_dir, set)
    data_pipe = dp.iter.FileLister(data_dir)
    data_pipe = data_pipe.open_files(mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()
    data_pipe = data_pipe.map(row_processor)
    return data_pipe

if __name__ == '__main__':
    data_pipe = build_datapipes()
    dl = DataLoader(
        dataset=data_pipe,
        batch_size=8,
        num_workers=2
    )

    first = next(iter(dl))
    labels, features = first['label'], first['data']
    print(f"Labels batch shape: {labels.size()}")
    print(f"Features batch shape: {features.size()}")

    n_sample = 0
    for row in iter(dl):
        n_sample += 1
    print(f'n_sample = {n_sample}')