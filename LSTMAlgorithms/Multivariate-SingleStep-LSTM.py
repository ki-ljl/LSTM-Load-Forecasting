# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/18 14:27
@Author ：KI 
@File ：Multivariate-SingleStep-LSTM.py
@Motto：Hungry And Humble

"""
from args import ms_args_parser
from util import train, test

LSTM_PATH = '../model/Multivariate-SingleStep-LSTM.pkl'


if __name__ == '__main__':
    args = ms_args_parser()
    flag = 'ms'
    train(args, LSTM_PATH, flag)
    test(args, LSTM_PATH, flag)
