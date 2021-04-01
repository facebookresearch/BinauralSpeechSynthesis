"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer import Trainer

dataset_dir = "/mnt/home/richardalex/tmp/bobatea/data/trainset"
artifacts_dir = "/mnt/home/richardalex/tmp/artifacts"

os.makedirs(artifacts_dir, exist_ok=True)

config = {
    "artifacts_dir": artifacts_dir,
    "learning_rate": 0.001,
    "newbob_decay": 0.5,
    "newbob_max_decay": 0.01,
    "batch_size": 32,
    "mask_beginning": 1024,
    "loss_weights": {"l2": 1.0, "phase": 0.01},
    "save_frequency": 10,
    "epochs": 100,
    "num_gpus": 4,
}

dataset = BinauralDataset(dataset_directory=dataset_dir, chunk_size_ms=200, overlap=0.5)

net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      wavenet_blocks=3,
                      layers_per_block=10,
                      wavenet_channels=64
                      )

print(f"receptive field: {net.receptive_field()}")
print(f"train on {len(dataset.chunks)} chunks")
print(f"number of trainable parameters: {net.num_trainable_parameters()}")
trainer = Trainer(config, net, dataset)
trainer.train()
