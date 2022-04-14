#!/bin/sh
read -p "Cuda: " cuda
read -p "Dataset: " dataset
read -p "Models: " models

declare -A depths=( ["resnet34"]="34" ["resnet18"]="18" ["vgg16"]="16" ["resnet50"]="50" )

for d in $dataset
do
for model in $models
do
  CUDA_VISIBLE_DEVICES=$cuda python main.py \
  --dataset $d \
  --arch $model \
  --depth "${depths[$model]}" \
  --lr 0.1 \
  --epochs 160 \
  --schedule 80 120 \
  --batch-size 256 \
  --test-batch-size 256 \
  --save ./EB/$d/$model \
  --momentum 0.9 \
  --sparsity-regularization
done
done
