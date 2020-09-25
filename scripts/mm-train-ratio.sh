#!/bin/bash
dataset=stb #stb or rhd
ratio=0.8 # 1 or 0.8 or 0.6 or 0.4 or 0.2
niter=1
optlevel=O1
gpus=0
port=9901
batchSize=3
IFS=',' read -ra nproc <<< "$gpus" # convert gpus into an array calls nproc
CUDA_VISIBLE_DEVICES=${gpus} python -m torch.distributed.launch\
       	--nproc_per_node=${#nproc[@]} \
	--master_port ${port}\
	train.py \
	    --dataroot ./datasets/${dataset}_dataset/train \
	    --name MMHand_${dataset}_${ratio} \
	    --lambda_GAN 5 \
	    --lambda_A 10  \
	    --lambda_B 10 \
	    --no_lsgan \
	    --n_layers 3 \
	    --batchSize ${batchSize} \
	    --no_flip \
	    --nThreads 4 \
	    --checkpoints_dir ./checkpoints \
	    --opt_level ${optlevel} \
	    --augmentation_ratio ${ratio}\
	    --augmentation_method GEN\
	    --dataset ${dataset}\
	    --save_latest_freq 400 \
	    --niter ${niter} --niter_decay 0\
	    --distributed

gpu=${nproc[0]}
ckp=MMHand_${dataset}_${ratio}
dataroot=./datasets/${dataset}_dataset
destination=./outputs/MM_${ratio}_${dataset}_dataset
if [ $ratio = 1 ]
then
    dataroot=${dataroot}/test
else
    dataroot=${dataroot}/train
fi
python aug.py ${ckp}\
    ${dataroot} \
    ${destination} \
    ${dataset} \
    ${ratio} \
    ${gpu}
