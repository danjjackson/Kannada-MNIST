import numpy as np
import torch
import random
import argparse
from model import *

import os
import importlib.util
import sys

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logging():
    return

def parse_args(cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        type=str,
        metavar='config.py',
        help='Python configuration file for the experiment'
    )
    parser.add_argument(
        '-n',
        '--num_epochs', 
        type=int, 
        default=5,
        help='Number of epochs to train the model for'
        )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='The path to the state dict of the model to be loaded'
    )
    return parser.parse_args(cmdline)

def load_config(filename):
    if not os.path.exists(filename):
        raise ValueError(f"The config sepcified does not exist: {filename}")
    if not os.path.isfile(filename):
        raise ValueError(f"The config specified is not a file: {filename}")
    if os.path.splitext(filename)[1] != ".py":
        raise ValueError("The config specified is not a python file")

    module_name = os.path.splitext(os.path.basename(filename))[0]
    if module_name in sys.modules:
        print(f"A module with the same name as '{module_name}' already exists")
        raise ImportError(f"Cannot import {module_name} as it already exists")

    config_dir = os.path.dirname(filename)
    if config_dir not in sys.path:
        sys.path.append(config_dir)
    spec = importlib.util.spec_from_file_location(module_name, filename)
    sys.modules[module_name] = importlib.util.module_from_spec(spec)
    config_module = spec.loader.load_module(module_name)
    pyconf = getattr(config_module, 'config')

    return pyconf

def create_model(model_name, model_kwargs):
    model_dict = {
        'ResNet': ResNet
    }
    if model_name in model_dict:
        return model_dict[model_name](**model_kwargs)
    else:
        assert False, f"Unknown model name \"{model_name}\". Available models are: {str(model_dict.keys())}"

def load_pretrained_model(model_name, model_hparams, path):
    model = create_model(model_name, model_hparams)
    model.load_state_dict(torch.load(path))
    return model