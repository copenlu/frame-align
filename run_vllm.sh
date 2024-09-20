#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=72:00:00

python src/models/vision/framing_vllm.py