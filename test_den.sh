#!/bin/bash

MODEL="unet43"
DATASET="galaxy10"
DEVICE="mps"

# --model-img-size 256 256 \
# Default parameters
python test.py \
    --data-dir "datasets/${DATASET}" \
    --dataset "torch/folder" \
    --train-split "train" \
    --val-split "validation" \
    --dataset-download \
    --topo\
    --model "${MODEL}" \
    --gp "avg" \
    --num-classes 10 \
    --img-size 128 \
    --in-chans 4 \
    --input-size 3 128 128 \
    --batch-size 32 \
    --sched multistep \
    --lr-base 0.1 \
    --epochs 20 \
    --decay-milestones 60 40 40 \
    --decay-rate 0.2 \
    --warmup-epochs 0 \
    --min-lr 0.0008 \
    --device "${DEVICE}" \
    --output "tests/${MODEL}_${DATASET}"\
    --output-img "tasks/denoising/galaxy10/restored/topo/0_2/"\
    --save-image \
    --denoising \
    --noise-level 0.2 \
    --mean 0.5 \
    --std 0.5 \
    --resume "tasks/denoising/galaxy10/models/topo/0_2/last.pth.tar"
