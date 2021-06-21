"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch as th
import torchaudio as ta

from src.models import BinauralNetwork
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./data/testset",
                    help="path to the test data")
parser.add_argument("--model_file",
                    type=str,
                    default="./outputs/binaural_network.net",
                    help="model file containing the trained binaural network weights")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./outputs",
                    help="directory to write binaural outputs to")
parser.add_argument("--blocks",
                    type=int,
                    default=3)
args = parser.parse_args()


def chunked_forwarding(net, mono, view):
    '''
    binauralized the mono input given the view
    :param net: binauralization network
    :param mono: 1 x T tensor containing the mono audio signal
    :param view: 7 x K tensor containing the view as 3D positions and quaternions for orientation (K = T / 400)
    :return: 2 x T tensor containing binauralized audio signal
    '''
    net.eval().cuda()
    mono, view = mono.cuda(), view.cuda()

    chunk_size = 480000  # forward in chunks of 10s
    rec_field = net.receptive_field() + 1000  # add 1000 samples as "safe bet" since warping has undefined rec. field
    rec_field -= rec_field % 400  # make sure rec_field is a multiple of 400 to match audio and view frequencies
    chunks = [
        {
            "mono": mono[:, max(0, i-rec_field):i+chunk_size],
            "view": view[:, max(0, i-rec_field)//400:(i+chunk_size)//400]
        }
        for i in range(0, mono.shape[-1], chunk_size)
    ]

    for i, chunk in enumerate(chunks):
        with th.no_grad():
            mono = chunk["mono"].unsqueeze(0)
            view = chunk["view"].unsqueeze(0)
            binaural = net(mono, view)["output"].squeeze(0)
            if i > 0:
                binaural = binaural[:, -(mono.shape[-1]-rec_field):]
            chunk["binaural"] = binaural

    binaural = th.cat([chunk["binaural"] for chunk in chunks], dim=-1)
    binaural = th.clamp(binaural, min=-1, max=1).cpu()
    return binaural


def compute_metrics(binauralized, reference):
    '''
    compute l2 error, amplitude error, and angular phase error for the given binaural and reference singal
    :param binauralized: 2 x T tensor containing predicted binaural signal
    :param reference: 2 x T tensor containing reference binaural signal
    :return: errors as a scalar value for each metric and the number of samples in the sequence
    '''
    binauralized, reference = binauralized.unsqueeze(0), reference.unsqueeze(0)

    # compute error metrics
    l2_error = L2Loss()(binauralized, reference)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized, reference)
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized, reference)

    return{
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "samples": binauralized.shape[-1]
    }


# binauralized and evaluate test sequence for the eight subjects and the validation sequence
test_sequences = [f"subject{i+1}" for i in range(8)] + ["validation_sequence"]

# initialize network
net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      wavenet_blocks=args.blocks,
                      layers_per_block=10,
                      wavenet_channels=64
                      )
net.load_from_file(args.model_file)

os.makedirs(f"{args.artifacts_directory}", exist_ok=True)

errors = []
for test_sequence in test_sequences:
    print(f"binauralize {test_sequence}...")

    # load mono input and view conditioning
    mono, sr = ta.load(f"{args.dataset_directory}/{test_sequence}/mono.wav")
    view = np.loadtxt(f"{args.dataset_directory}/{test_sequence}/tx_positions.txt").transpose().astype(np.float32)
    view = th.from_numpy(view)

    # sanity checks
    if not sr == 48000:
        raise Exception(f"sampling rate is expected to be 48000 but is {sr}.")
    if not view.shape[-1] * 400 == mono.shape[-1]:
        raise Exception(f"mono signal is expected to have 400x the length of the position/orientation sequence.")

    # binauralize and save output
    binaural = chunked_forwarding(net, mono, view)
    ta.save(f"{args.artifacts_directory}/{test_sequence}.wav", binaural, sr)

    # compute error metrics
    reference, sr = ta.load(f"{args.dataset_directory}/{test_sequence}/binaural.wav")
    errors.append(compute_metrics(binaural, reference))

# accumulate errors
sequence_weights = np.array([err["samples"] for err in errors])
sequence_weights = sequence_weights / np.sum(sequence_weights)
l2_error = sum([err["l2"] * sequence_weights[i] for i, err in enumerate(errors)])
amplitude_error = sum([err["amplitude"] * sequence_weights[i] for i, err in enumerate(errors)])
phase_error = sum([err["phase"] * sequence_weights[i] for i, err in enumerate(errors)])

# print accumulated errors on testset
print(f"l2 (x10^3):     {l2_error * 1000:.3f}")
print(f"amplitude:      {amplitude_error:.3f}")
print(f"phase:          {phase_error:.3f}")

