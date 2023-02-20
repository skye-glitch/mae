# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc_sequential as misc
#from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_sequential

from engine_pretrain_sequential import train_one_epoch
import util.lr_sched as lr_sched
import deepspeed


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Move it to deepspeed setup! Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--fp32', default=False, action='store_true',
                        help='do not use mixed precision')
    parser.add_argument('--pipeline_parallel_size', default=4, type=int, help='number of stages, 0 if not using pipeline')



    return parser


def main(args):
    misc.init_distributed_mode(args)

    if torch.distributed.get_rank() == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

    #device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'ILSVRC2012_img_train'), transform=transform_train)
    if torch.distributed.get_rank() == 0:
        print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=(num_tasks//args.pipeline_parallel_size)
                if args.pipeline_parallel_size > 0 else
                num_tasks, rank=(global_rank//args.pipeline_parallel_size) 
                if args.pipeline_parallel_size > 0 else global_rank, shuffle=True
        )
        if torch.distributed.get_rank() == 0:
            print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_ = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size*args.accum_iter if 
            args.pipeline_parallel_size > 0 else args.batch_size,
        #num_workers=args.num_workers,
        num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    net = models_mae_sequential.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if args.pipeline_parallel_size > 0:
        #init pipeline module
        from deepspeed.pipe import PipelineModule
        #layers = [module for module in net.modules()]
        def join_layers(vision_model):
            layers = [*vision_model]
            return layers
        #todo: speed and loss without pipeline
        net = PipelineModule(layers=join_layers(net),
                            num_stages=args.pipeline_parallel_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(net, args.weight_decay)
   
    
    engine, optimizer, data_loader_train, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=param_groups,
        training_data=dataset_train)

    
    eff_batch_size = (args.batch_size * args.accum_iter * 
        misc.get_world_size() // args.pipeline_parallel_size)  if args.pipeline_parallel_size > 0 else (args.batch_size * args.accum_iter * misc.get_world_size())
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr
   
    if torch.distributed.get_rank() == 0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    misc.load_model(args=args, engine=engine)
    if torch.distributed.get_rank() == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    step_per_epoch = len(dataset_train)//eff_batch_size
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train_.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            engine,
            data_loader_train_,
            step_per_epoch,
            optimizer,
            epoch,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 15 == 0 or epoch + 1 == args.epochs):
            output_dir = Path(args.output_dir)
            epoch_name = str(epoch)
            checkpoint_path = output_dir
            to_save = {
                'epoch': epoch,
                'args': args,
            }
            engine.save_checkpoint(checkpoint_path, epoch_name, client_state=to_save)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if torch.distributed.get_rank() == 0:
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    deepspeed.init_distributed(dist_backend='nccl')
    # Include DeepSpeed configuration arguments
    args = deepspeed.add_config_arguments(args)
    args = args.parse_args()
    
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
