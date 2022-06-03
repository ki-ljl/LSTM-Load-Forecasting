# -*- coding:utf-8 -*-
"""
@Time：2022/04/04 23:10
@Author：KI
@File：multivariate_multi_step.py
@Motto：Hungry And Humble
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mm_args_parser
from util import train, test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/multivariate_multi_step.pkl'


if __name__ == '__main__':
    args = mm_args_parser()
    flag = 'mm'
    train(args, LSTM_PATH, flag)
    test(args, LSTM_PATH, flag)
