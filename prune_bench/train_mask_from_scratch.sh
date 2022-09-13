#!/bin/bash

read -p "Cuda: " cuda
read -p "Dataset: " d
read -p "Model: " model
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed

prunes="SNIP GraSP SynFlow ERK Rand iterSNIP PHEW"
density="0.05 0.10 0.20 0.40"

for dense in $density; do
for prune in $prunes; do
    echo $prune
    CUDA_VISIBLE_DEVICES=$cuda python3 main.py --model $model --data $d \
    --decay-schedule constant \
    --seed $seed \
    --optimizer sgd\
    --prune $prune\
    --sparse \
    --density $dense
done
done
