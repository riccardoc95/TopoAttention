#!/bin/bash

MODEL="densenet121small"
DATASET="mnist"
DEVICE="mps"

# Default parameters
python train.py \
    --data-dir datasets \
    --dataset "torch/${DATASET}" \
    --train-split "train" \
    --val-split "validation" \
    --dataset-download \
    --model "${MODEL}" \
    --gp "avg" \
    --num-classes 10 \
    --img-size 28 \
    --in-chans 1 \
    --input-size 1 28 28 \
    --batch-size 512 \
    --sched multistep \
    --lr-base 0.1 \
    --epochs 100 \
    --decay-milestones 20 20 20 \
    --decay-rate 0.2 \
    --warmup-epochs 0 \
    --min-lr 0.0008 \
    --device "${DEVICE}" \
    --output "output/${MODEL}_${DATASET}"