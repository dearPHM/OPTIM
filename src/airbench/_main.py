import os
from time import time
import pickle
import torch

import torchvision
from dataloader import CifarLoader
from model import make_net, train


############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))


logging_columns_list = ['run   ', 'epoch', 'train_loss',
                        'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


if __name__ == "__main__":
    from hyperparameters import hyp
    # print_columns(logging_columns_list, is_head=True)

    # Set Params
    batch_size = hyp['opt']['batch_size']
    epochs = 9.9  # Hardcoded
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    # un-decoupled learning rate for PyTorch SGD
    lr = 11.5 / kilostep_scale  # Hardcoded
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']
    label_smoothing = hyp['opt']['label_smoothing']
    whiten_bias_epochs = hyp['opt']['whiten_bias_epochs']

    tta_level = hyp['net']['tta_level']

    # Make Model
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    scaling_factor = hyp['net']['scaling_factor']
    model = make_net(widths, batchnorm_momentum, scaling_factor)

    # Set Dataset
    # train_loader = CifarLoader(
    #     'cifar10',
    #     train=True, batch_size=batch_size, aug=hyp['aug'])
    # test_loader = CifarLoader(
    #     'cifar10',
    #     train=False, batch_size=2000)
    train_loader = CifarLoader(torchvision.datasets.CIFAR10('./data/cifar10', download=True, train=True),
                               train=True, batch_size=batch_size, aug=hyp['aug'])
    test_loader = CifarLoader(torchvision.datasets.CIFAR10('./data/cifar10', download=True, train=False),
                              train=False, batch_size=2000)

    # Run
    # train('warmup')
    _, _, acc_collect = train(
        # run,
        model,
        train_loader, test_loader,
        batch_size, epochs, momentum, lr, wd, lr_biases,
        label_smoothing, whiten_bias_epochs,
        tta_level
    )
    print("accs:", acc_collect)

    # Save
    log_dir = os.path.join('logs', str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pkl')
    print(os.path.abspath(log_path))
    with open(log_path, 'wb') as f:
        pickle.dump([None, acc_collect], f)
