# -*- coding:utf-8 -*-
"""
@Time：2022/04/15 15:35
@Author：KI
@File：models.py
@Motto：Hungry And Humble
"""
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
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
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (5 * 30, 64)
        pred = self.linear(output)  # pred()
        # print('pred=', pred.shape)
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]

        return pred
