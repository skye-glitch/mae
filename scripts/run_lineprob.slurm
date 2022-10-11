#!/bin/bash
#SBATCH -J lineprob           # Job name
#SBATCH -o log/lineprob.o%j       # Name of stdout output file
#SBATCH -e log/lineprob.e%j       # Name of stderr error file
#SBATCH -p gpu-a100            # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 6
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH -A Deep-Learning-at-Sca       # Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu



NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)

./scripts/copy_and_extract.sh /work/07980/sli4/ls6/data/imagenet-1k.tar /tmp/imagenet


mpiexec.hydra -np $NNODES -ppn 1 ./scripts/run_lineprob.sh --batch_size 170 \
    --model vit_huge_patch14 \
    --finetune /work/07980/sli4/ls6/mae/output_dir/mae_pretrain_vit_huge.pth \
    --epochs 50 \
    --blr 0.1  \
    --weight_decay 0.0 \
    --dist_eval --data_path /tmp/imagenet \
    --accum_iter 16 \
    --output_dir ./output_dir_lineprob 
