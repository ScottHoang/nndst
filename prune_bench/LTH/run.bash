#!/bin.bash
read -p "Cuda: " cuda
read -p "Dataset: " d
read -p "Model: " model
read -p "seed: " seed

density="0.05 0.10 0.20 0.40"

for dense in $density; do
    python3 main.py --arch_type $model --data $d --seed $seed --density $dense --gpu $cuda
done
