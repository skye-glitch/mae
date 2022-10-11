#!/bin/bash
#SBATCH -J evaluate           # Job name
#SBATCH -o log/evaluate.o%j       # Name of stdout output file
#SBATCH -e log/evaluate.e%j       # Name of stderr error file
#SBATCH -p gpu-a100            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH -A Deep-Learning-at-Sca       # Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)


mpiexec.hydra -np $NNODES -ppn 1 ./scripts/evaluate.sh  \
--eval  --model vit_huge_patch14 --batch_size 16 --data_path /work/07980/sli4/ls6/data \
--resume /work/07980/sli4/ls6/mae/output_dir/mae_finetuned_vit_huge.pth
