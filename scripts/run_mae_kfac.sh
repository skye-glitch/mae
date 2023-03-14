#!/bin/bash
LOCAL_RANK=$PMI_RANK

CMD="main_pretrain_kfac.py $@"

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE


GPU_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi


PRELOAD="module load python3/3.9; "
PRELOAD+="source /work2/07980/sli4/frontera/venv/bin/activate;"
PRELOAD+="export OMP_NUM_THREADS=4 ; "


LAUNCHER="python -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=4 \
--node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK"

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"

echo $FULL_CMD 

eval $FULL_CMD &

wait