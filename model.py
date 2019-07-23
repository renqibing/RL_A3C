import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mycode.utils import norm_col_init, weights_init, weights_init_mlp


def weights_init(m):
    name = m.__class__.__name__
    if name.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_out = weight_shape[0]
        fan_in = weight_shape[1]
        w_bound = np.sqrt((6./ (fan_in+fan_out)))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        # print('initialized {}'.format(name))


def normalized_columns_initializer(weights,std):
    data = torch.randn(weights.size())
    data *= std / torch.sqrt(data.pow(2).sum(1, keepdim = True))
    return data


class A3CMLP(nn.Module):
    def __init__(self, num_inputs, action_space, stackedtimes):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.ml = stackedtimes * 128
        self.lstmcell = nn.LSTMCell(self.ml, 128)

        self.criticlayer = nn.Linear(128, 1)
        num_outputs = action_space.shape[0]
        self.actorlayer1 = nn.Linear(128, num_outputs)
        self.actorlayer2 = nn.Linear(128, num_outputs)


        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        self.actorlayer1.weight.data = norm_col_init(
            self.actorlayer1.weight.data, 0.01)
        self.actorlayer1.bias.data.fill_(0)
        self.actorlayer2.weight.data = norm_col_init(
            self.actorlayer2.weight.data, 0.01)
        self.actorlayer2.bias.data.fill_(0)
        self.criticlayer.weight.data = norm_col_init(
            self.criticlayer.weight.data, 1.0)
        self.criticlayer.bias.data.fill_(0)

        self.lstmcell.bias_ih.data.fill_(0)
        self.lstmcell.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        x = x.view(1, self.ml)
        hx, cx = self.lstmcell(x, (hx, cx))
        x = hx

        return self.criticlayer(x), F.softsign(self.actorlayer1(x)), self.actorlayer2(x), (hx, cx)


class PolicyNet(nn.Module):

    def __init__(self, state_space, hidden_space1 = 400, hidden_space2 = 300, output_space = 1):
        super().__init__()
        self.fc1 = nn.Linear(state_space, hidden_space1)
        self.fc2 = nn.Linear(hidden_space1, hidden_space2)
        self.fc3 = nn.Linear(hidden_space2, output_space)
        self.relu = nn.ReLU()
        self.apply(weights_init)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight, 0.01
        )


    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out


class ValueNet(nn.Module):

    def __init__(self,input_space,hidden_space1,hidden_space2,output_space):
        super().__init__()
        self.fc1 = nn.Linear(input_space, hidden_space1)
        self.fc2 = nn.Linear(hidden_space1, hidden_space2)
        self.fc3 = nn.Linear(hidden_space2, output_space)
        self.relu = nn.ReLU()
        self.apply(weights_init)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight, 1.0
        )

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out