# -*- coding:utf-8 -*-
"""
@Time：2022/04/04 23:10
@Author：KI
@File：Multivariate-MultiStep-LSTM.py
@Motto：Hungry And Humble
"""
import os
from args import mm_args_parser
from util import train, test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/model/Multivariate-MultiStep_LSTM.pkl'


if __name__ == '__main__':
    args = mm_args_parser()
    flag = 'mm'
    train(args, LSTM_PATH, flag)
    test(args, LSTM_PATH, flag)
