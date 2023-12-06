#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="DELMC2"
python3 -m pip install -r requirements.txt
echo

NB_PATH="fhnw-del-mc2.ipynb"
python3 -m papermill $NB_PATH $NB_PATH -k 'python3'

echo "Done!"