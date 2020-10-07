!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=9901 \
    train.py --gpu_ids 0 --distributed \
    --dataroot datasets/STB_cropped/train \
    --name None_stb_1 \
    --dataset_mode hpm \
    --model hpm \
    --dataset STB \
    --batch_size 20 \
    --num_threads 6 \
    --lr 0.0002 \
    --input_nc 21 \
    --print_freq 1000 \
    --save_latest_freq 4000 \
    --opt_level O0 \
    --niter 50 \
    --niter_decay 0 \
    --augmentation_method None \
    --augmentation_ratio 1  \

CUDA_VISIBLE_DEVICES=0 python test.py \
    --gpu_ids 0 \
    --dataroot datasets/STB_cropped//test \
    --dataset STB \
    --model hpm \
    --input_nc 21 \
    --name None_stb_1


