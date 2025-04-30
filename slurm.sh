#!/bin/bash

#SBATCH --job-name="train"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu-a100-small
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=runs/train.%j.out
#SBATCH --error=runs/train.%j.err

#source ./.venv/bin/activate

module load 2024r1
module load openmpi
module load py-torch
module load py-pandas
#module load gcc
#module load py-six

#srun torchrun --standalone --nproc_per_node=2 scripts/main.py
python scripts/main.py

