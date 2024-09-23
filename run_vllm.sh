#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00

python src/models/text/text_framing_sample_vllm.py\
    --model_code mistralai/Mistral-7B-Instruct-v0.2 \
    --data_file /projects/copenlu/data/arnav/frame-align/raw/2023-24/topic_samples.csv