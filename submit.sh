#!/bin/bash
#SBATCH --job-name=spert
#SBATCH --account=pfaendtner
#SBATCH --partition=gpu-2080ti
#SBATCH --gpus=2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10 
#SBATCH --time=0-10:00:00 
#SBATCH --mem=40G 

python ./spert.py train --config configs/ams_train.conf
