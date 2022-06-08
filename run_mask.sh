
read -p "Cuda: " cuda
read -p "Dataset: " dataset
read -p "Model: " model
read -p "batchsize: " batchsize
read -p "epochs: " epochs
read -p "seed: " seed
read -p "masks: " masks
read -p "mask-type: " mask_type # iou/struct/unstruct

cd ./Early-Bird-Tickets
for mask in $masks
do
    CUDA_VISIBLE_DEVICES=$cuda python main-train-pruned.py \
    --dataset $dataset \
    --arch $model \
    --depth 1 \
    --lr 0.1 \
    --epochs $epochs \
    --schedule 80 120 \
    --batch-size $batchsize \
    --test-batch-size 256 \
    --save ./results_macro_masks/$dataset/$model/$mask_type/$mask \
    --momentum 0.9 \
    --sparsity-regularization \
    --seed $seed \
    --mask ./marco_mask/$dataset/$model/$mask_type\_$mask\.pth.tar
done
