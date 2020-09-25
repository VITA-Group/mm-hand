#!/bin/bash
########### Options ##############
dataset=stb # Select either stb or rhd dataset to train and generate
ratio=1     # Select the training data ratio. 1 is training on 100% all data, 0.8 is 80% and so
            #   on. Options are 1 or 0.8 or 0.6 or 0.4 or 0.2
niter=100   # Number of training iteration
optlevel=O1 # Mix precision mode options. This allow for faster training time.
            #   O0: training on fp32
            #   O1: training on fp32/fp16
            #   O2: training on fp32/fp16 (not recommended)
            #   O3: training on fp16 (not recommended)
gpus=0      # to have more than one gpu, do 0,1,2
port=9901
batchSize=3 # batch size
IFS=',' read -ra nproc <<< "$gpus" # convert gpus into an array calls nproc


###### Training Script ###########
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


##### Generating Script ##########
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
