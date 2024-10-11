#!/bin/bash

pwd=$(cd $(dirname $0); pwd)
echo pwd: $pwd

backbone = cnndetection
# Remove directories containing "ipynb" files
find ${pwd}/dataset -type d -name "*ipynb*" -exec rm -r {} +

# Run the training script with specified parameters
python train.py \
--name adof-$backbone- \
--dataroot ${pwd}/dataset/ForenSynths_train \
--num_thread 8 \
--classes car,cat,chair,horse \
--batch_size 64 \
--delr_freq 5 \
--loss_freq 400 \
--lr 0.0002 \
--niter 10 \
--blur_prob 0 --blur_sig 0 --jpg_prob 0 --jpg_method cv2 --jpg_qual 100 \
--backbone $backbone 
