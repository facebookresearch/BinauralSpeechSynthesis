"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse

from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./data/trainset",
                    help="path to the training data")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./outputs",
                    help="directory to write model files to")
parser.add_argument("--num_gpus",
                    type=int,
                    default=4,
                    help="number of GPUs used during training")
parser.add_argument("--blocks",
                    type=int,
                    default=3)
args = parser.parse_args()

config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 0.001,
    "newbob_decay": 0.5,
    "newbob_max_decay": 0.01,
    "batch_size": 32,
    "mask_beginning": 1024,
    "loss_weights": {"l2": 1.0, "phase": 0.01},
    "save_frequency": 10,
    "epochs": 100,
    "num_gpus": args.num_gpus,
}

os.makedirs(config["artifacts_dir"], exist_ok=True)

dataset = BinauralDataset(dataset_directory=args.dataset_directory, chunk_size_ms=200, overlap=0.5)

net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      wavenet_blocks=args.blocks,
                      layers_per_block=10,
                      wavenet_channels=64
                      )

print(f"receptive field: {net.receptive_field()}")
print(f"train on {len(dataset.chunks)} chunks")
print(f"number of trainable parameters: {net.num_trainable_parameters()}")
trainer = Trainer(config, net, dataset)
trainer.train()
