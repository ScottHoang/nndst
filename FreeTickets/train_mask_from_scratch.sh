#!/bin/bash

read -p "Cuda: " cuda
read -p "Masks dir " masks_dir
read -p "Dataset: " d
read -p "Model: " model
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed

for string in "$masks_dir/*pth"; do
    for mask in $string; do
        echo $mask
        CUDA_VISIBLE_DEVICES=$cuda python3 main_train_DST.py --model $model --data $d \
        --decay-schedule constant \
        --seed $seed \
        --mask_path $mask \
        --lr 0.001 \
        --start_from_init \
        --nolrsche \
        --optimizer adam
    done
done
