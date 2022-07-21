#!/bin/sh
#
#SBATCH --job-name="unixcoder finetune"
#SBATCH --partition=gpu
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --account=research-eemcs-st

module load 2022r2
module load cuda/11.3
module load python/3.8.12
module load py-pip
#python -m pip install --user -r ../requirements.txt
python train.py /scratch/tovandam/datasets/python-dataset
