# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/18 14:27
@Author ：KI 
@File ：Multivariate-SingleStep-LSTM.py
@Motto：Hungry And Humble

"""
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import os
from scipy.interpolate import make_interp_spline
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import matplotlib.pyplot as plt
from data_process import setup_seed

setup_seed(20)
MAX, MIN = 0, 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_PATH = 'model/Multivariate-SingleStep-LSTM.pkl'


def load_data(file_name):
    global MAX, MIN
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    MAX = np.max(df[columns[1]])
    MIN = np.min(df[columns[1]])
    df[columns[1]] = (df[columns[1]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq(file_name, B):
    print('data processing...')
    data = load_data(file_name)
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []
    for i in range(len(data) - 24):
        train_seq = []
        train_label = []
        for j in range(i, i + 24):
            x = [load[j]]
            for c in range(2, 8):
                x.append(data[j][c])
            train_seq.append(x)
        train_label.append(load[i + 24])
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    Dtr = seq[0:int(len(seq) * 0.7)]
    Dte = seq[int(len(seq) * 0.7):len(seq)]

    train_len = int(len(Dtr) / B) * B
    test_len = int(len(Dte) / B) * B
    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Dtr, Dte


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        # print('output.size=', output.size())
        # print(self.batch_size * seq_len, self.hidden_size)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)
        pred = self.linear(output)  # pred()
        # print('pred=', pred.shape)
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred


def LSTM_train(name, b):
    Dtr, Dte = nn_seq(file_name=name, B=b)
    input_size, hidden_size, num_layers, output_size = 7, 64, 1, 1
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=b).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # training
    epochs = 30
    for i in range(epochs):
        cnt = 0
        for (seq, label) in Dtr:
            cnt += 1
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cnt % 100 == 0:
                print('epoch', i, ':', cnt - 100, '~', cnt, loss.item())
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, LSTM_PATH)


def get_mape(x, y):
    """
    :param x:true
    :param y:predict
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))


def test(name, b):
    global MAX, MIN
    Dtr, Dte = nn_seq(file_name=name, B=b)
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers, output_size = 7, 64, 1, 1
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=b).to(device)
    model.load_state_dict(torch.load(LSTM_PATH)['model'])
    model.eval()
    print('predicting...')
    for (seq, target) in Dte:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array([y]), np.array([pred])
    y = (MAX - MIN) * y + MIN
    pred = (MAX - MIN) * pred + MIN
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y.T[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred.T[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    name = 'anqiudata.csv'
    B = 30
    LSTM_train(name, B)
    test(name, B)
