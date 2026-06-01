#!/bin/bash
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=training_log.txt

cd /home/20251020/ECG
python3 -u train_1unet.py
