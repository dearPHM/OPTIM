#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--byzantines', type=int, default=0,
                        help="number of Byzantine users who submit zero weights: Z")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_users', type=int, default=21,
                        help="number of users (nodes): K")
    parser.add_argument('--frac', type=float, default=0.1,  # 10%
                        help='the fraction of clients: C')

    # fedasync arguments
    parser.add_argument('--stale', type=int, default=4,
                        help='max staleness (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='mixing hyperparameter (default: 0.6)')

    # BRAIN arguments
    parser.add_argument('--diff', type=float, default=0.55,
                        help='the franction related to quorum (default: 0.55 => 21*0.55=11 nodes)')
    parser.add_argument('--window', type=int, default=4,
                        help='window size for moving averaging (default: 4) (>= 2)')
    parser.add_argument('--threshold', type=float, default=0.125,
                        help='accuracy threshold to ignore (default: 0.125)')
    parser.add_argument('--advanced_threshold', type=bool, default=False,
                        help='advanced threshold tactic based on congestion control method')
    parser.add_argument('--score_byzantines', type=int, default=0,
                        help="number of Byzantine users who submit random score: SZ")
    parser.add_argument('--drift', type=int, default=0,
                        help="number of users who are drifted from global model")
    parser.add_argument('--optim', type=bool, default=False,
                        help="enabling optim method")
    parser.add_argument('--history', type=int, default=0,
                        help="0 -> no history / 1-> history with streak / 2-> no rejection, average score")
    parser.add_argument('--maxqueue', type=int, default=20,
                        help="max length of queue for registering updates")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # parser.add_argument('--norm', type=str, default='batch_norm',
    #                     help="batch_norm, layer_norm, or None")
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--lr', type=float, default=11.5,  # 9.0 for single SGD
                        help='learning rate (default: 11.5)')

    # other arguments
    # parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
    # to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--epochs', type=int, default=200,  # Rounds
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=float, default=9.9,  # 10
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1024,
                        help="local batch size: B")
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--optimizer', type=str, default='sgd', help="type \
    #                     of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID (and unequal).')
    # parser.add_argument('--unequal', type=int, default=0,
    #                     help='whether to use unequal data splits for  \
    #                     non-i.i.d setting (use 0 for equal splits)')
    # parser.add_argument('--stopping_rounds', type=int, default=10,
    #                     help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args
