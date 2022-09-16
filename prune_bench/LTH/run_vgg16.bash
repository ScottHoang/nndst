#!/bin.bash
cuda=0
d="cifar10"
models="vgg16"
seeds="1" #2 3"
density="0.01" # 0.05 0.10 0.20 0.40"
result_dir="/home/sliu/project_space/PaI_graph/LTH/"

for dense in $density; do
    for seed in $seeds; do
        for model in $models; do
            echo $model $seed $dense
            python3 main.py --arch_type $model --data $d --seed $seed --density $dense --gpu $cuda \
		    --result_dir $result_dir
        done
    done
done
