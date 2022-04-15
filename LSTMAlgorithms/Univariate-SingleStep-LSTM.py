# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/18 14:27
@Author ：KI 
@File ：Univariate-SingleStep-LSTM.py
@Motto：Hungry And Humble

"""
from util import train, test
from args import us_args_parser
from data_process import setup_seed

setup_seed(20)
LSTM_PATH = '../model/Univariate-SingleStep-LSTM.pkl'


if __name__ == '__main__':
    args = us_args_parser()
    flag = 'us'
    train(args, LSTM_PATH, flag)
    test(args, LSTM_PATH, flag)
