#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=28G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=72:00:00

vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype half --api-key framing & > vllm_out.txt
sleep 5m
python src/models/text/text_framing_sample_vllm.py

kill %1