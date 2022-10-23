model_params = {
    "num_classes": 10,
    "act_fn_name": "relu"
    }

optimizer_params = {
    "lr": 1e-3,
    "weight_decay": 1e-4
    }

config = {
    'model_name': 'SimpleConvNet',
    'model_hparams': model_params,
    'batch_size': 256,
    'optimizer_name': 'Adam',
    'optimizer_hparams': optimizer_params
}