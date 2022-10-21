model_hparams={
    'dropout': 0.1,
    'num_classes': 10, 
    'num_blocks': [3,3,3], 
    'c_hidden': [16,32,64], 
}
training_params = {
    'learning_rate': 1e-5,
    'weight_decay': 1e-5,
    'batch_size':256
}

config = {
    'model_name': 'ResNet',
    **training_params,
    **model_hparams,
}