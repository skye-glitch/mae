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

import util.misc_kfac as misc_kfac
from util.misc_kfac import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain_kfac import train_one_epoch

#todo new import
import kfac
import logging
#from apex.optimizers import FusedLAMB


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

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
    parser.add_argument(
        '--optim',
        default='adamW',
        type=str,
        help='optimizer',
        choices=['adamW', 'lamb'],
    )
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
    #todo, add kfac group
    kfac_group = parser.add_argument_group('KFAC Parameters')
    kfac_group.add_argument(
        '--inv-update-steps',
        type=int,
        default=10,
        help='iters between updating second-order information',
    )
    kfac_group.add_argument(
        '--factor-update-steps',
        type=int,
        default=1,
        help='iters between update kronecker factors',
    )
    kfac_group.add_argument(
        '--firstInv',
        type=int,
        default=0,
        help='first epoch to do inverse',
    )
    parser.add_argument(
        '--kfac-inv-method',
        action='store_true',
        default=False,
        help='Use inverse KFAC update instead of eigen (default False)',
    )
    kfac_group.add_argument(
        '--factor-decay',
        type=float,
        default=0.95,
        help='alpha value for factor accumulation',
    )
    kfac_group.add_argument(
        '--damping',
        type=float,
        default=0.003,
        help='damping factor',
    )
    kfac_group.add_argument(
        '--kl-clip',
        type=float,
        default=0.001,
        help='KL clip',
    )
    kfac_group.add_argument(
        '--skip-layers',
        nargs='+',
        type=str,
        default=['embedding', 'decoder'],
        help='layers to skip KFAC registration for',
    )
    parser.add_argument(
        '--kfac-colocate-factors',
        action='store_true',
        default=True,
        help='Compute A and G for a single layer on the same worker. ',
    )
    kfac_group.add_argument(
        '--kfac_strategy',
        choices=['mem-opt', 'hybrid-opt', 'comm-opt'],
        default='comm-opt',
        help='distribution strategy for KFAC computations',
    )
    parser.add_argument(
        '--kfac-grad-worker-fraction',
        type=float,
        default=0.25,
        help='Fraction of workers to compute the gradients '
        'when using HYBRID_OPT (default: 0.25)',
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    parser.add_argument(
        '--loss_bound',
        type=float,
        default='inf',
        help='bounding loss',
    )

    return parser


def main(args):
    misc_kfac.init_distributed_mode(args)

    if torch.distributed.get_rank() == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc_kfac.get_rank()
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
        num_tasks = misc_kfac.get_world_size()
        global_rank = misc_kfac.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
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

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        #num_workers=args.num_workers,
        #num_workers=0,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    if torch.distributed.get_rank() == 0:
        print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc_kfac.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if torch.distributed.get_rank() == 0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95)) 
    if torch.distributed.get_rank() == 0:
        print(optimizer)
    loss_scaler = NativeScaler()

     # todo: add preconditioner
    preconditioner = None
    use_kfac = True if args.inv_update_steps > 0 else False
    if use_kfac:
        grad_worker_fraction: kfac.enums.DistributedStrategy | float
        if args.kfac_strategy == 'comm-opt':
            grad_worker_fraction = kfac.enums.DistributedStrategy.COMM_OPT
        elif args.kfac_strategy == 'mem-opt':
            grad_worker_fraction = kfac.enums.DistributedStrategy.MEM_OPT
        elif args.kfac_strategy == 'hybrid-opt':
            grad_worker_fraction = args.kfac_grad_worker_fraction
        else:
            raise ValueError(
                f'Unknown KFAC Comm Method: {args.kfac_strategy}',
            )
        preconditioner = kfac.preconditioner.KFACPreconditioner(
            model,
            factor_update_steps=args.factor_update_steps,
            inv_update_steps=args.inv_update_steps,
            damping=args.damping,
            factor_decay=args.factor_decay,
            kl_clip=args.kl_clip,
            lr=lambda x: optimizer.param_groups[0]['lr'],
            grad_worker_fraction=grad_worker_fraction,
            skip_layers=args.skip_layers,
            loglevel=logging.INFO,
            allreduce_bucket_cap_mb=25,
            colocate_factors=args.kfac_colocate_factors,
            compute_method=kfac.enums.ComputeMethod.INVERSE
            if args.kfac_inv_method
            else kfac.enums.ComputeMethod.EIGEN,
            grad_scaler=args.grad_scaler if 'grad_scaler' in args else None,
            firstInv=args.firstInv,
        )
        if torch.distributed.get_rank() == 0:
            print(f'Preconditioner config:\n{preconditioner}')

    misc_kfac.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, preconditioner=preconditioner, loss_scaler=loss_scaler)

   

    if torch.distributed.get_rank() == 0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, preconditioner, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 15 == 0 or epoch + 1 == args.epochs):
            misc_kfac.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, 
                preconditioner=preconditioner, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc_kfac.is_main_process():
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
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
