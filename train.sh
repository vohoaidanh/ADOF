#!/bin/bash

# Remove directories containing "ipynb" files
find /workspace/datasets -type d -name "*ipynb*" -exec rm -r {} +

# Run the training script with specified parameters
python train.py \
--name adof-fastvit_sa24- \
--dataroot /workspace/datasets/ForenSynths_train \
--num_thread 2 \
--classes car,cat,chair,horse \
--batch_size 32 \
--delr_freq 5 \
--loss_freq 400 \
--lr 0.0002 \
--niter 10 \
--backbone efficientnet_b0
