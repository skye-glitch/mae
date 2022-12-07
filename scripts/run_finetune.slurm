#!/bin/bash
#SBATCH -J finetune           # Job name
#SBATCH -o log/finetune.o%j       # Name of stdout output file
#SBATCH -e log/finetune.e%j       # Name of stderr error file
#SBATCH -p gpu-a100            # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 6
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH -A Deep-Learning-at-Sca       # Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --dependency=524908



NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)

./scripts/copy_and_extract.sh /work/07980/sli4/ls6/data/imagenet-1k.tar /tmp/imagenet

# mpiexec.hydra -np $NNODES -ppn 1 ./scripts/run_finetune.sh --batch_size 86 \
#     --model vit_base_patch16 \
#     --finetune /work/07980/sli4/ls6/mae/output_dir/mae_pretrain_vit_base.pth \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval \
#     --data_path /tmp/imagenet \
#     --accum_iter 2 \
#     --output_dir ./output_dir_finetune
     #   --data=/work/07980/sli4/ls6/data

# mpiexec.hydra -np $NNODES -ppn 1 ./scripts/run_finetune.sh --batch_size 42 \
#     --model vit_huge_patch14 \
#     --finetune /work/07980/sli4/ls6/mae/output_dir/mae_pretrain_vit_huge.pth \
#     --epochs 50 \
#     --blr 1e-3 --layer_decay 0.75 \
#     --weight_decay 0.05 --drop_path 0.3 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path /tmp/imagenet \
#     --accum_iter 4 \
#     --output_dir ./output_dir_finetune_huge \
#     --resume /work/07980/sli4/ls6/mae/output_dir_finetune_huge/checkpoint-24.pth

mpiexec.hydra -np $NNODES -ppn 1 ./scripts/run_finetune.sh --batch_size 42 \
    --batch_size 42 \
    --model vit_large_patch16 \
    --finetune /work/07980/sli4/ls6/mae/output_dir_large/checkpoint-799.pth \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path /tmp/imagenet \
    --accum_iter 4 \
    --output_dir ./output_dir_finetune_large


