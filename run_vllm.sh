#!/bin/bash
#SBATCH --job-name=t_frame   # Job name
#SBATCH --array=0-11
#SBATCH --gres=gpu:a100:1  # Number of GPUs (per node)
#SBATCH --cpus-per-task=2     # 8 CPUs per task
#SBATCH --mem=20GB            # 30GB of memory
#SBATCH --time=10-00:00:00        # runtime
#SBATCH --partition=gpu


base_dir="/projects/frame_align/data/raw/2023-2024"
directories=($(ls -d ${base_dir}/*))

month_dir=${directories[$SLURM_ARRAY_TASK_ID]}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "processing file: ${month_dir}/datawithtopiclabels.csv"

python src/models/text/text_framing_vllm.py --data_file "${month_dir}/datawithtopiclabels.csv" --output_dir "/projects/frame_align/data/annotated/text/" --model_code mistralai/Mistral-7B-Instruct-v0.3