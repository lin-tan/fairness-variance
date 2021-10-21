#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python3 train.py --save_dir origin_0 --log_dir origin_0 --batch_size 32 --num_epochs 60 --learning_rate 0.0001 --seed 1
mv models/origin_0 adv/

CUDA_VISIBLE_DEVICES=0 python3 ae_adv_train.py --save_dir 6 --log_dir 6 --layer generated_image --adv_lambda 5.0 --batch_size 16 --beta 0.1 --batch_balanced --num_epochs 80 --learning_rate 0.00001 --autoencoder_finetune --adv_on --resume --seed 1
CUDA_VISIBLE_DEVICES=0 python3 ae_adv_train.py --save_dir 6_75 --log_dir 6_75 --layer generated_image --adv_lambda 5.0 --batch_size 16 --beta 0.1 --batch_balanced --num_epochs 160 --learning_rate 0.00001 --resume --finetune --checkpoint checkpoint_75.pth.tar --seed 1 --variance_logging
CUDA_VISIBLE_DEVICES=0 python3 natural_leakage.py --num_rounds 1 --num_epochs 100 --learning_rate 0.00005 --batch_size 128 --no_image --seed 1
CUDA_VISIBLE_DEVICES=0 python3 ae_adv_attacker.py --exp_id generated_image_5.0_0.1_6_75 --adv_on --layer generated_image --adv_capacity 300 --adv_lambda 5 --learning_rate 0.00005 --num_epochs 100 --batch_size 128 --best_model --num_rounds 1 --seed 1