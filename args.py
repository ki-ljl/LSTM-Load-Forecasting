# -*- coding:utf-8 -*-
"""
@Time：2022/04/15 15:30
@Author：KI
@File：args.py
@Motto：Hungry And Humble
"""
import argparse
import torch


# Multivariate-MultiStep-LSTM
def mm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args


# Multivariate-SingleStep-LSTM
def ms_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

    args = parser.parse_args()

    return args


# Univariate-SingleStep-LSTM
def us_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args
