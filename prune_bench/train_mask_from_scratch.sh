#!/bin/bash

read -p "Cuda: " cuda
read -p "Dataset: " d
read -p "Model: " model
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed

prunes="SNIP GraSP SynFlow ERK Rand"
density=0.01

for prune in $prunes; do
    echo $prune
    CUDA_VISIBLE_DEVICES=$cuda python3 main.py --model $model --data $d \
    --decay-schedule constant \
    --seed $seed \
    --lr 0.001 \
    --nolrsche \
    --optimizer adam\
    --prune $prune\
    --sparse \
    --density $density
done
