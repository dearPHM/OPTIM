hyp = {
    'opt': {
        # 'train_epochs': 9.9,
        'batch_size': 1024,
        # learning rate per 1024 examples
        # 'lr': 11.5,  # for FL
        # 'lr': 9.0,  # for Single (suitable for many epochs)
        'momentum': 0.85,
        # weight decay per 1024 examples (decoupled from learning rate)
        'weight_decay': 0.0153,
        # scales up learning rate (but not weight decay) for BatchNorm biases
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        # how many epochs to train the whitening layer bias before freezing
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
        'tta_level': 2,
    },
}
