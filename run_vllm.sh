#!/bin/bash
#SBATCH --job-name=sy_framing   # Job name
#SBATCH --array=0
#SBATCH --gres=gpu:a40:1 
#SBATCH --cpus-per-task=4     # 8 CPUs per task
#SBATCH --mem=40GB            # 30GB of memory
#SBATCH --time=10:00:00        # runtime
#SBATCH --partition=gpu

# base_dir="/projects/frame_align/data/raw/2023-2024"
# directories=($(ls -d ${base_dir}/*))

# month_dir=${directories[$SLURM_ARRAY_TASK_ID]}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "processing file: ${month_dir}/datawithtopiclabels.csv"

# --exclude hendrixgpu03fl,hendrixgpu04fl,hendrixgpu15fl,hendrixgpu21fl,hendrixgpu22fl
# python src/models/vision/vision_framing_local.py.py --data_file "${month_dir}/datawithtopiclabels.csv" --output_dir "/projects/frame_align/data/srishti_part1/" --model_code mistralai/Mistral-7B-Instruct-v0.3
# python src/models/vision/vision_framing_local.py --data_file "data_pixtral_llava/sample_data.csv" --dir_name "data_pixtral_llava/output" --model_name llava-hf/llava-1.5-13b-hf
python src/models/vision/vision_framing_local.py --data_file "data_pixtral_llava/val_set_full.csv" --output_dir "data_pixtral_llava/output" --model_name "mistralai/Pixtral-12B-2409"