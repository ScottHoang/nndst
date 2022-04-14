#!/bin/bash

read -p "Cuda: " cuda
read -p "Dataset: " dataset
read -p "Models: " models
read -p "batchsize: " batchsize
read -p "epochs: " epochs
echo Generate Random Weights

python generate_init_weights.py "69" "./common_models/random_weights" "10,100,100"
echo Completed

cd ./Early-Bird-Tickets

for d in $dataset
do
    for model in $models
        do
      CUDA_VISIBLE_DEVICES=$cuda python main.py \
      --dataset $d \
      --arch $model \
      --depth 1 \
      --lr 0.1 \
      --epochs $epochs \
      --schedule 80 120 \
      --batch-size $batchsize \
      --test-batch-size 256 \
      --save ./EB/$d/$model \
      --momentum 0.9 \
      --sparsity-regularization
        done
done


cd ../

cd ./FreeTickets
for d in $dataset
do
for model in $models
do
CUDA_VISIBLE_DEVICES=$cuda python3 main_EDST.py --sparse --model $model --data $d --nolrsche \
    --decay-schedule constant --seed 17 --epochs-explo 0 \
    --model-num 10 \
    --sparse-init ERK \
    --update-frequency 1000 --batch-size $batchsize --death-rate 0.5 --large-death-rate 0.8 \
    --growth gradient --death magnitude --redistribution none --epochs $batchsize --density 0.2
done
done

cd ../
