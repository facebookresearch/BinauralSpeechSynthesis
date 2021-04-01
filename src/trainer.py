"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import time
import torch as th
from torch.utils.data import DataLoader

from src.utils import NewbobAdam
from src.losses import L2Loss, PhaseLoss


class Trainer:
    def __init__(self, config, net, dataset):
        '''
        :param config: a dict containing parameters
        :param net: the network to be trained, must be of type src.utils.Net
        :param dataset: the dataset to be trained on
        '''
        self.config = config
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)
        gpus = [i for i in range(config["num_gpus"])]
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = NewbobAdam(weights,
                                    net,
                                    artifacts_dir=config["artifacts_dir"],
                                    initial_learning_rate=config["learning_rate"],
                                    decay=config["newbob_decay"],
                                    max_decay=config["newbob_max_decay"])
        self.l2_loss = L2Loss(mask_beginning=config["mask_beginning"])
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.total_iters = 0
        # switch to training mode
        self.net.train()

    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def train(self):
        for epoch in range(self.config["epochs"]):
            t_start = time.time()
            loss_stats = {}
            for data in self.dataloader:
                loss_new = self.train_iteration(data)
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            t_end = time.time()
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            print(f"epoch {epoch+1} " + loss_str + "        " + time_str)
            # Save model
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch+1))
                print("Saved model")
        # Save final model
        self.save()

    def train_iteration(self, data):
        '''
        one optimization step
        :param data: tuple of tensors containing mono, binaural, and quaternion data
        :return: dict containing values for all different losses
        '''
        # forward
        self.optimizer.zero_grad()

        mono, binaural, quats = data
        mono, binaural, quats = mono.cuda(), binaural.cuda(), quats.cuda()
        prediction = self.net.forward(mono, quats)
        l2 = self.l2_loss(prediction["output"], binaural)
        phase = self.phase_loss(prediction["output"], binaural)
        intermediate_binaural = th.cat([binaural] * len(prediction["intermediate"]), dim=1)
        intermediate_prediction = th.cat(prediction["intermediate"], dim=1)
        intermediate_l2 = self.l2_loss(intermediate_prediction, intermediate_binaural)
        intermediate_phase = self.phase_loss(intermediate_prediction, intermediate_binaural)

        loss = (l2 + intermediate_l2) * self.config["loss_weights"]["l2"] + \
               (phase + intermediate_phase) * self.config["loss_weights"]["phase"]

        # update model parameters
        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return {
            "l2": l2,
            "phase": phase,
            "intermediate_l2": intermediate_l2,
            "intermediate_phase": intermediate_phase,
            "accumulated_loss": loss,
        }
