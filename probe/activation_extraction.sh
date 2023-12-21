#!/bin/bash
#SBATCH --job-name=activation-extraction
#SBATCH --output=/nas/ucb/henrypapadatos/Sycophancy/probe/%j.out
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1

source /nas/ucb/henrypapadatos/Miniconda3/bin/activate
conda activate LAT

cd /nas/ucb/henrypapadatos/Sycophancy/probe

srun echo "Hello World from" `hostname`
srun echo "Current file name is " "$PWD"
srun activation_extraction.py