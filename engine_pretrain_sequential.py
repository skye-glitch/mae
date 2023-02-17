# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc_sequential as misc
import util.lr_sched as lr_sched
from torch.autograd import Variable


def train_one_epoch(engine,
                    data_loader: Iterable, 
                    # data_iter,
                    tot_steps,
                    optimizer: torch.optim.Optimizer,
                    epoch: int, 
                    log_writer=None,
                    args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    #for data_iter_step in range(tot_steps):#(samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #if data_iter_step % accum_iter == 0:
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        #todo: speed and loss without pipeline
        #samples = samples.cuda(misc.get_rank(), non_blocking=True)
        #todo: speed and loss without pipeline
        loss = engine.train_batch()
        #loss = engine(samples)
        #runs backpropagation
        #engine.backward(loss)
        #weight update
        #engine.step()
        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}