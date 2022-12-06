#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A SuperBERT
#PBS -M sli@tacc.utexas.edu
#PBS -l filesystems=home:grand
#PBS -m be 




./scripts/evaluate.sh  \
--eval  --model vit_huge_patch14 --batch_size 16 --data_path /work/07980/sli4/ls6/data \
--resume /work/07980/sli4/ls6/mae/output_dir/mae_finetuned_vit_huge.pth
