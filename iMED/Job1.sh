#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH --qos=gpulab02
#SBATCH -J recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --gres=gpu:1
source activate recon
python train.py
