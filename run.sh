#!/bin/bash
#SBATCH --job-name=v_frame   # Job name
#SBATCH --array=0-11
#SBATCH --gres=gpu:titanrtx:1  # Number of GPUs (per node)
#SBATCH --partition=gpu    # Partition to submit to
#SBATCH --cpus-per-task=2     # 8 CPUs per task
#SBATCH --mem=20GB            # 30GB of memory
#SBATCH --time=10-00:00:00        # runtime


base_dir="/projects/frame_align/data/raw/2023-2024"
directories=($(ls -d ${base_dir}/*))

current_dir=${directories[$SLURM_ARRAY_TASK_ID]}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "processing $current_dir"
echo "processin file: ${current_dir}/datawithtopiclabels.csv"

# python3 src/models/vision/vision_framing.py --data_file "${current_dir}/datawithtopiclabels.csv" --dir_name "${current_dir}" 

python3 src/models/vision/vision_framing.py --dir_name "${current_dir}" 
