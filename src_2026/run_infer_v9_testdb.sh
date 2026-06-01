#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=Lenovo6
#SBATCH --output=/shared_storage/iulia.orvas/paper/fECG_approx/logs/infer_v9_testdb.log

export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

cd /shared_storage/iulia.orvas/paper/fECG_approx/src_2026
python -u infer_testdb_v9.py
