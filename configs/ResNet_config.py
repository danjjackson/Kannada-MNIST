model_params={
    'num_classes': 10, 
    'num_blocks': [3,3,3], 
    'c_hidden': [16,32,64],
    'act_fn_name': 'relu'
}

optimizer_params = {
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
}

config = {
    'model_name': 'ResNet',
    'model_hparams': model_params,
    'batch_size': 256,
    'optimizer_name': 'SGD',
    'optimizer_hparams': optimizer_params
}