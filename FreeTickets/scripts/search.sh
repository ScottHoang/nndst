#!/bin/sh
read -p "Cuda: " cuda
read -p "Dataset: " dataset
read -p "Models: " models


for d in $dataset
do
for model in $models
do
  CUDA_VISIBLE_DEVICES=$cuda python3 main_EDST.py --sparse --model $model --data $d --nolrsche --decay-schedule constant \
  --seed 17  --epochs-explo 150 --model-num 3 --sparse-init ERK --update-frequency 1000 --batch-size 128 --death-rate 0.5 \
  --large-death-rate 0.8 --growth gradient --death magnitude --redistribution none --epochs 450 --density 0.2
done
done