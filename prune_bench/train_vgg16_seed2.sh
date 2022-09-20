#!/bin/bash
#SBATCH --job-name=vgg16_seed2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o vgg16_seed2.out

source /home/sliu/miniconda3/etc/profile.d/conda.sh
source activate slak


cuda=0
d='cifar10'
model='vgg16'
seed='2'
prunes="SNIP GraSP SynFlow ERK Rand iterSNIP" #PHEW"
density="0.01 0.05 0.1 0.2 0.4"
skip_if_exist=true # if directory exist, skip if true otherwise generate another instance within
save_dir="/home/sliu/project_space/PaI_graph/LTH/vgg16_seed2/"
for dense in $density; do
for prune in $prunes; do
    if [ ! -d "$save_dir/density_$dense/$d/$model/$prune/$seed" ] || [ $skip_if_exist != true ] ;
    then
	echo "$save_dir/density_$dense/$d/$model/$prune/$seed"
	CUDA_VISIBLE_DEVICES=$cuda python3 main.py --model $model --data $d \
	--decay-schedule constant \
	--seed $seed \
	--optimizer sgd\
	--prune $prune\
	--sparse \
	--density $dense \
	--save_dir $save_dir
    fi
done
done
