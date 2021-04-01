"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.hyperconv import HyperConvBlock
from src.warping import GeometricTimeWarper, MonotoneTimeWarper
from src.utils import Net


class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)

    def _transmitter_mouth(self, view):
        # offset between tracking markers and real mouth position in the dataset
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        # make sure zero-padded values are set to non-zero values (else scipy raises an exception)
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = th.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        # offset between tracking markers and ears in the dataset
        left_ear_offset = th.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, -0.08, -0.22])
        right_ear_offset = th.Tensor([0, 0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, 0.08, -0.22])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        displacement = th.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        return self.warper(th.cat([mono, mono], dim=1), self._3d_displacements(view))


class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()

    def neural_warpfield(self, view, seq_length):
        warpfield = view
        for layer in self.layers:
            warpfield = F.pad(warpfield, pad=[1, 0])
            warpfield = F.relu(layer(warpfield))
        warpfield = self.linear(warpfield)
        warpfield = F.interpolate(warpfield, size=seq_length)
        return warpfield

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        geometric_warpfield = self.geometric_warper._warpfield(view, mono.shape[-1])
        neural_warpfield = self.neural_warpfield(view, mono.shape[-1])
        warpfield = geometric_warpfield + neural_warpfield
        # ensure causality
        warpfield = -F.relu(-warpfield)
        warped = self.neural_warper(th.cat([mono, mono], dim=1), warpfield)
        return warped


class HyperConvWavenet(nn.Module):
    def __init__(self, z_dim, channels=64, blocks=3, layers_per_block=10, conv_len=2):
        super().__init__()
        self.layers = []
        self.rectv_field = 1
        for b in range(blocks):
            for l in range(layers_per_block):
                self.layers += [HyperConvBlock(channels, channels, z_dim, kernel_size=conv_len, dilation=2**l)]
                self.rectv_field += self.layers[-1].receptive_field() - 1
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, z):
        '''
        :param x: input signal as a B x channels x T tensor
        :param z: weight-generating input as a B x z_dim z K tensor (K = T / 400)
        :return: x: output signal as a B x channels x T tensor
                 skips: skip signal for each layer as a list of B x channels x T tensors
        '''
        skips = []
        for layer in self.layers:
            x, skip = layer(x, z)
            skips += [skip]
        return x, skips

    def receptive_field(self):
        return self.rectv_field


class WaveoutBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size=1)
        self.first.weight.data.uniform_(-np.sqrt(6.0 / channels), np.sqrt(6.0 / channels))
        self.second = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, x):
        x = th.sin(self.first(x))
        return self.second(x)


class BinauralNetwork(Net):
    def __init__(self,
                 view_dim=7,
                 warpnet_layers=4,
                 warpnet_channels=64,
                 wavenet_blocks=3,
                 layers_per_block=10,
                 wavenet_channels=64,
                 model_name='binaural_network',
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.warper = Warpnet(warpnet_layers, warpnet_channels)
        self.input = nn.Conv1d(2, wavenet_channels, kernel_size=1)
        self.input.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))
        self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels, wavenet_blocks, layers_per_block)
        self.output_net = nn.ModuleList([WaveoutBlock(wavenet_channels)
                                        for _ in range(wavenet_blocks*layers_per_block)])
        if self.use_cuda:
            self.cuda()

    def forward(self, mono, view):
        '''
        :param mono: the input signal as a B x 1 x T tensor
        :param view: the receiver/transmitter position as a B x 7 x T tensor
        :return: out: the binaural output produced by the network
                 intermediate: a two-channel audio signal obtained from the output of each intermediate layer
                               as a list of B x 2 x T tensors
        '''
        warped = self.warper(mono, view)
        x = self.input(warped)
        _, skips = self.hyperconv_wavenet(x, view)
        # collect output and skips after each layer
        x = []
        for k in range(len(skips), 0, -1):
            y = th.mean(th.stack(skips[:k], dim=0), dim=0)
            y = self.output_net[k-1](y)
            x += [y]
        x += [warped]
        return {"output": x[0], "intermediate": x[1:]}

    def receptive_field(self):
        return self.hyperconv_wavenet.receptive_field()
