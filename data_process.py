# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 20:11
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    """
    :return:
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/data.csv'
    df = pd.read_csv(path, encoding='gbk')
    df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multivariate-MultiStep-LSTM data processing.
def nn_seq_mm(seq_len, B, num):
    print('data processing...')
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size, step_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=num)

    return Dtr, Val, Dte, m, n


# Multivariate-SingleStep-LSTM data processing.
def nn_seq_ms(seq_len, B):
    print('data processing...')
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(data[j][c])
                train_seq.append(x)
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)

    return Dtr, Val, Dte, m, n


# Univariate-SingleStep-LSTM data processing.
def nn_seq_us(seq_len, B):
    print('data processing...')
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B)
    Val = process(val, B)
    Dte = process(test, B)

    return Dtr, Val, Dte, m, n


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
