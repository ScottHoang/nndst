#!/bin/bash

read -p "Cuda: " cuda
read -p "Dataset: " dataset
read -p "Models: " models
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed
read -p "sparsity: " sparsity
read -p "output graph dir: " dir
echo Generate Random Weights

python generate_init_weights.py $seed "./common_models/random_weights" "10,100,1000"
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
      --sparsity-regularization \
      --seed $seed
	done
done


cd ../

epochs=$(($epochs + 150))
cd ./FreeTickets
for d in $dataset
do
for model in $models
do
CUDA_VISIBLE_DEVICES=$cuda python3 main_EDST.py --sparse --model $model --data $d --nolrsche \
    --decay-schedule constant --seed $seed --epochs-explo 150 \
    --model-num 10 \
    --sparse-init ERK \
    --density $sparsity \
    --update-frequency 1000 --batch-size $batchsize --death-rate 0.5 --large-death-rate 0.8 \
    --growth gradient --death magnitude --redistribution none --epochs $epochs \
    --seed $seed
done
done

cd ../

eb_sparse=$((1.0 - $sparsity))

python generate_graphs.py "./Early-Bird-Tickets/EB" "$dir/EB/" $eb_sparse "1"

python generate_graphs.py "./FreeTickets/results/" "$dir/FreeTickets/" "$sparsity" "0"
