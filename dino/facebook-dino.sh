#!/bin/bash
OUTPUT_PATH=/s/kbalak18/Facebook-Dino-master/pose_kb1/output/

python3 main_dino.py --arch vit_small --num_workers=8 --local_crops_number=0 --global_crops_scale 0.14 1. --out_dim 32768 --use_bn_in_head False --epochs 201 --batch_size_per_gpu 12 --use_fp16 False --data_path /s/hpc-datasets/ml-datasets/imagenet --output_dir ${OUTPUT_PATH}
