#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=72:00:00

vllm serve llava-hf/llava-1.5-7b-hf --dtype half --api-key framing & > vllm_vision.log

python src/models/vision/vlm_framing_vllm.py

kill %1