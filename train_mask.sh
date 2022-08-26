#!/bin/bash

read -p "Cuda: " cuda
read -p "Masks dir " masks_dir
read -p "Dataset: " dataset
read -p "Model: " model
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed

for string in "$masks_dir/*pth.tar"; do
    for mask in $string; do
        echo $mask
        echo $model
        # CUDA_VISIBLE_DEVICES=$cuda python3 main_train_DST.py --model $model --data $d --nolrsche \
        # --decay-schedule constant \
        # --seed $seed \
        # --mask_path $mask
    done
done

# cd ./FreeTickets
# for d in $dataset
# do
# for model in $models
# do
# CUDA_VISIBLE_DEVICES=$cuda python3 main_train_DST.py --model $model --data $d --nolrsche \
#     --decay-schedule constant \
#     --seed $seed \
#     --mask_path results/cifar10/masks/10h24m17s_on_Aug_07_2022/3000_mask.pth.tar
#     done
# done

# cd ../
