#!/bin/bash

#SBATCH --job-name="pre_train"

#SBATCH --workdir=/home/nct01/nct01068/Lab2

#SBATCH --output=pre_train.out

#SBATCH --error=pre_train.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=40:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python vgg16_pretrained_trainable_train.py
