#!/bin/bash
#SBATCH --output=/shared_storage/iulia.orvas/paper/fECG_approx/logs/infer_all_testdb.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

export LD_LIBRARY_PATH=/shared_storage/iulia.orvas/miniconda3/envs/ecg/lib:/shared_storage/iulia.orvas/miniconda3/lib:$LD_LIBRARY_PATH

source /shared_storage/iulia.orvas/miniconda3/etc/profile.d/conda.sh
conda activate ecg

cd /shared_storage/iulia.orvas/paper/fECG_approx/src_2026
python -u infer_testdb_all.py
