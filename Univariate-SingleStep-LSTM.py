# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/18 14:27
@Author ：KI 
@File ：Univariate-SingleStep-LSTM.py
@Motto：Hungry And Humble

"""
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import make_interp_spline

from data_process import nn_seq, device, get_mape
LSTM_PATH = 'model/Univariate-SingleStep-LSTM.pkl'


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
        seq_len = input_seq.shape[1]  # (5, 30)
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, 1)  # (5, 30, 1)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        # print('output.size=', output.size())
        # print(self.batch_size * seq_len, self.hidden_size)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)
        pred = self.linear(output)  # pred()
        # print('pred=', pred.shape) # 150*1
        pred = pred.view(self.batch_size, seq_len, -1)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


def LSTM_train(name, b):
    Dtr, Dte, MAX, MIN = nn_seq(file_name=name, B=b)
    input_size, hidden_size, num_layers, output_size = 1, 32, 2, 1
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=b).to(device)
    # model.load_state_dict(torch.load(LSTM_PATH)['model'])
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer.load_state_dict(torch.load(LSTM_PATH)['optimizer'])
    model.train()
    # training
    epochs = 5  # 10 15
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


def test(name, b):
    Dtr, Dte, MAX, MIN = nn_seq(file_name=name, B=b)
    pred = []
    y = []
    print('loading model...')
    input_size, hidden_size, num_layers, output_size = 1, 32, 2, 1
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=b).to(device)
    model.load_state_dict(torch.load(LSTM_PATH)['model'])
    model.eval()
    print('predicting...')
    for (seq, target) in Dte:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        seq_len = seq.shape[1]
        seq = seq.view(model.batch_size, seq_len, 1)  # (5, 30, 1)
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
    B = 50
    LSTM_train(name, B)
    test(name, B)
