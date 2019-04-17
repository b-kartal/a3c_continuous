from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init


class A3C_CONV(torch.nn.Module):
    def __init__(self, num_inputs, action_space, terminal_prediction, reward_prediction):
        super(A3C_CONV, self).__init__()
        self.conv1 = nn.Conv1d(num_inputs, 32, 3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv1d(32, 64, 2, stride=1, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv1d(64, 64, 1, stride=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(1600, 128)
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.terminal_aux_head = None
        if terminal_prediction:  # this comes with the arg parser
            self.terminal_aux_head = nn.Linear(128, 1)  # output a single prediction

        self.reward_aux_head = None
        if reward_prediction:
            self.reward_aux_head = nn.Linear(128, 1)  # output a single estimate of reward prediction

        self.apply(weights_init)
        lrelu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv1.weight.data.mul_(lrelu_gain)
        self.conv2.weight.data.mul_(lrelu_gain)
        self.conv3.weight.data.mul_(lrelu_gain)
        self.conv4.weight.data.mul_(lrelu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # new added parts for auxiliary tasks within the network
        if terminal_prediction:
            self.terminal_aux_head.weight.data = norm_col_init(self.terminal_aux_head.weight.data, 1.0)
            self.terminal_aux_head.bias.data.fill_(0)

        if reward_prediction:
            self.reward_aux_head.weight.data = norm_col_init(self.reward_aux_head.weight.data, 1.0)
            self.reward_aux_head.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        if self.terminal_aux_head is None:
            terminal_prediction = None
        else:
            terminal_prediction = torch.sigmoid(self.terminal_aux_head(x))

        if self.reward_aux_head is None:
            reward_prediction = None
        else:
            reward_prediction = self.reward_aux_head(x)

        return self.critic_linear(x), F.softsign(self.actor_linear(x)), self.actor_linear2(x), (hx, cx), terminal_prediction, reward_prediction # last two outputs are auxiliary tasks
