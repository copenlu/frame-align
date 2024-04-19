#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=280G
#SBATCH --gres=gpu
#SBATCH --time=72:00:00

python src/models/MFC.py
