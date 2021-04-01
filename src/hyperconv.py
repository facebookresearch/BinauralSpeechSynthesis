"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class HyperConv(nn.Module):
    def __init__(self, input_size, ch_in, ch_out, kernel_size, dilation=1):
        '''
        HyperConv implements a temporal convolution that has different convolution weights for each time step.
        :param input_size: (int) dimension of the weight generating input variable
        :param ch_in: (int) number of input channels of the temporal convolution
        :param ch_out: (int) number of output channels of the temporal convolution
        :param kernel_size: (int) kernel size of the temporal convolution
        :param dilation: (int) dilation of the temporal convolution
        '''
        super().__init__()
        weight_regressor_hidden_size = 32
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(weight_regressor_hidden_size, ch_in * ch_out * kernel_size, kernel_size=1)
        )
        self.bias_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(weight_regressor_hidden_size, ch_out, kernel_size=1)
        )
        # initialize weights such that regressed weights are distributed in a suitable way for sine activations
        self.weight_model[0].weight.data.zero_()
        self.weight_model[0].bias.data.zero_()
        self.weight_model[-1].bias.data.uniform_(-np.sqrt(6.0/(self.ch_in*self.kernel_size)),
                                                 np.sqrt(6.0/(self.ch_in*self.kernel_size)))

    def forward(self, x, z):
        '''
        :param x: the input signal as a B x ch_in x T tensor
        :param z: the weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: a B x ch_out x T tensor as the result of the hyper-convolution
        '''
        B = x.shape[0]
        assert x.shape[-1] % z.shape[-1] == 0
        # padding
        padding = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, [padding, 0])
        # linearize input by appending receptive field in channels
        start, end = padding, x.shape[-1]
        x = th.cat([x[:, :, start-i*self.dilation:end-i*self.dilation] for i in range(self.kernel_size)], dim=1)
        # rearrange input to blocks for matrix multiplication
        x = x.permute(0, 2, 1).contiguous().view(x.shape[0] * z.shape[-1], x.shape[-1]//z.shape[-1], x.shape[1])
        # compute weights and bias
        weight = self.weight_model(z).view(B, self.ch_in * self.kernel_size, self.ch_out, z.shape[-1])
        weight = weight.permute(0, 3, 1, 2).contiguous().view(B * z.shape[-1], self.ch_in * self.kernel_size, self.ch_out)
        bias = self.bias_model(z).view(B, self.ch_out, z.shape[-1])
        bias = bias.permute(0, 2, 1).contiguous().view(B * z.shape[-1], self.ch_out)
        # compute result of dynamic convolution
        y = th.bmm(x, weight)
        y = y + bias[:, None, :]
        y = y.view(B, -1, self.ch_out).permute(0, 2, 1).contiguous()
        return y


class HyperConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, z_dim, kernel_size, dilation=1):
        '''
        :param ch_in: (int) input channels
        :param ch_out: (int) output channels
        :param z_dim: (int) dimension of the weight-generating input
        :param kernel_size: (int) size of the filter
        :param dilation: (int) dilation
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = HyperConv(z_dim, ch_in, ch_out, kernel_size, dilation)
        self.residual = nn.Conv1d(ch_out, ch_out, kernel_size=1)
        self.residual.weight.data.uniform_(-np.sqrt(6.0/ch_out), np.sqrt(6.0/ch_out))
        self.skip = nn.Conv1d(ch_out, ch_out, kernel_size=1)
        self.skip.weight.data.uniform_(-np.sqrt(6.0/ch_out), np.sqrt(6.0/ch_out))
        if not ch_in == ch_out:
            self.equalize_channels = nn.Conv1d(ch_in, ch_out, kernel_size=1)
            self.equalize_channels.weight.data.uniform_(-np.sqrt(6.0 / ch_in), np.sqrt(6.0 / ch_in))

    def forward(self, x, z):
        '''
        :param x: input signal as a B x ch_in x T tensor
        :param z: weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: output: B x ch_out x T tensor as layer output
                 skip: B x ch_out x T tensor as skip connection output
        '''
        assert x.shape[-1] % z.shape[-1] == 0
        y = self.conv(x, z)
        y = th.sin(y)
        # residual and skip
        residual = self.residual(y)
        if not self.ch_in == self.ch_out:
            x = self.equalize_channels(x)
        skip = self.skip(y)
        return (residual + x) / 2, skip

    def receptive_field(self):
        return (self.kernel_size - 1) * self.dilation + 1
