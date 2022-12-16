#!/bin/bash
#SBATCH -G 1
#SBATCH -n 10
#SBATCH --time=55:00:00

nvidia-smi
python run.py data=$1 train.NUM_EPOCHS=15 train.TRAIN_BATCH_SIZE=5 train.GRAD_ACCUM_STEPS=3 train.VALIDATION_BATCH_SIZE=10
