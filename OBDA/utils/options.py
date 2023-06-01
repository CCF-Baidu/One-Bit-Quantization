#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default= 1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=256, help="local batch size: B")
    parser.add_argument('--local_bs2', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--M', type=int, default=10, help='total # of devices')
    parser.add_argument('--N', type=int, default=5, help='# of BS antennas')  #
    parser.add_argument('--L', type=int, default=40, help='RIS Size')
    parser.add_argument('--bit', type=int, default=1, help='phase shift resolution')  # np.array([1, 2, 3, np.inf]
    parser.add_argument('--U', type=int, default=40, help='# of subchannel（subcarrier if s=1）')  #
    parser.add_argument('--nit', type=int, default=100, help='I_max,# of maximum SCA loops')
    parser.add_argument('--Jmax', type=int, default=50, help='# of maximum Gibbs Outer loops')
    parser.add_argument('--threshold', type=float, default=1e-2, help='epsilon,SCA early stopping criteria')
    parser.add_argument('--tau', type=float, default=1, help=r'\tau, the SCA regularization term')
    parser.add_argument('--verbose', type=int, default=0, help=r'whether output or not')
    parser.add_argument('--P0', type=float, default=0.10, help='transmit budget P_0')
    parser.add_argument('--SNR', type=float, default=95.0, help='noise variance/0.1W in dB')
    parser.add_argument('--set', type=int, default=2, help=r'=1 if concentrated devices+ euqal dataset;\
                                =2 if two clusters + unequal dataset')  #
    parser.add_argument('--epislon', type=float, default=1e-8,
                        help='\epislon, the DC')  # SCA regularization term for Algorithm 1
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='femnist', help="name of dataset")
    parser.add_argument('--iid', default='True', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--lr-scheduler', action='store_true', help='whether using lr scheduler')

    # quan
    parser.add_argument('--mode', type=str, default='AWGN', help="Tx mode")
    # snr
    parser.add_argument('--snr', type=int, default=-10, help="SNR")
    # delta: channel estimation error
    parser.add_argument('--delta', type=float, default= 0.01, help="Delta")
    # g_th
    parser.add_argument('--thd', type=float, default= 4.2, help="THD")

    args = parser.parse_args()
    return args
