#!/bin/sh
#
#SBATCH --job-name="create tasks"
#SBATCH --partition=compute
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --account=research-eemcs-st

module load 2022r2
npx run createLineCompletionTasks.ts /scratch/tovandam/datasets/javascript-dataset
