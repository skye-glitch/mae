
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE


GPU_PER_NODE=$(lspci | grep NVIDIA | wc -l)

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

CMD="main_finetune.py $@"


PRELOAD="source /work/07980/sli4/ls6/conda/etc/profile.d/conda.sh ; "
PRELOAD+="conda activate torch-1.11;"
PRELOAD+="source switch-cuda.sh;"
PRELOAD+="source switch-cuda.sh 11.3;"

if [ $NNODES -eq 1 ]; then
    LAUNCHER="python -m torch.distributed.launch \
      --nproc_per_node=$GPU_PER_NODE"
else
    LAUNCHER="python -m torch.distributed.launch "
    LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=$GPU_PER_NODE \
    --node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK"
fi


# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"

eval $FULL_CMD &

wait




