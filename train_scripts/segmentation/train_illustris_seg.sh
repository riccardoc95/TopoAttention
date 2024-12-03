#!/bin/bash

MODEL="unet11"
DATASET="illustris-segment"
DEVICE="mps"

# --model-img-size 256 256 \
# Default parameters
CUDA_VISIBLE_DEVICES=1 python train.py \
    --data-dir "datasets/" \
    --dataset "astro/${DATASET}" \
    --train-split "train" \
    --val-split "validation" \
    --dataset-download \
    --model "${MODEL}" \
    --gp "avg" \
    --num-classes 2 \
    --img-size 128 \
    --in-chans 1 \
    --input-size 1 128 128 \
    --batch-size 64 \
    --sched multistep \
    --lr-base 0.1 \
    --epochs 20 \
    --decay-milestones 5 5 5 \
    --decay-rate 0.2 \
    --warmup-epochs 0 \
    --min-lr 0.0008 \
    --device "${DEVICE}" \
    --output "output/${MODEL}_${DATASET}"\
    --save-image \
    --segmentation