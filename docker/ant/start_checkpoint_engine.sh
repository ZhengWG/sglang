#!/bin/bash

NPROC_PER_NODE=8
MASTER_PORT=29500
# Extract tp-size from ENGINE_EXTRA_ARGS
INFERENCE_PARALLEL_SIZE=$(echo "$ENGINE_EXTRA_ARGS" | grep -oP '(?<=--tp-size )\d+')
ENDPOINT=http://localhost:$PORT
MODEL_DIR=/home/admin/model/
LOG_FILE=/home/admin/logs/update.log
UPDATE_METHOD=broadcast
UPDATE_SCRIPT=/root/checkpoint_engine/update.py


# Run checkpoint engine with torchrun
torchrun \
    --nproc-per-node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $UPDATE_SCRIPT \
    --endpoint $ENDPOINT \
    --update-method $UPDATE_METHOD \
    --checkpoint-path $MODEL_DIR \
    --inference-parallel-size $INFERENCE_PARALLEL_SIZE \
    >> $LOG_FILE 2>&1
