#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
srun hostname
ml py-scikit-learn/1.0.2_py39
pip3 install --user -r requirements.txt
ruse python3 run_rf_xgboost.py 1992 1997 Fluazinam


python3 run_rf_xgboost.py 1992 1997 --reverse
