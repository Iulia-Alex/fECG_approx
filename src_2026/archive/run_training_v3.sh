#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=Lenovo2
#SBATCH --output=/shared_storage/iulia.orvas/paper/fECG_approx/logs/movement_CUNet_128x400_peakw.log

# Use conda's libstdc++ instead of the system's (system lacks GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

cd /shared_storage/iulia.orvas/paper/fECG_approx/src_2026
python -u train_movement_v3.py
