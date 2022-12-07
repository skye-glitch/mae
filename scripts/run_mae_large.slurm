#!/bin/bash
#SBATCH -J mae_large          # Job name
#SBATCH -o log/mae.o%j       # Name of stdout output file
#SBATCH -e log/mae.e%j       # Name of stderr error file
#SBATCH -p gpu-a100            # Queue (partition) name
#SBATCH -N 4               # Total # of nodes (must be 1 for serial)
#SBATCH -n 12
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -A Deep-Learning-at-Sca       # Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu



NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)

./scripts/copy_and_extract.sh /work/07980/sli4/ls6/data/imagenet-1k.tar /tmp/imagenet


mpiexec.hydra -np $NNODES -ppn 1 ./scripts/run_mae.sh --batch_size 171 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /tmp/imagenet \
    --accum_iter 2 \
    --output_dir ./output_dir_large \
    --resume /work/07980/sli4/ls6/mae/output_dir_large/checkpoint-750.pth \
    --log_dir ./output_dir_large
     #   --data=/work/07980/sli4/ls6/data
