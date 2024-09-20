#!/bin/bash
#SBATCH --job-name=sy_frame   # Job name
#SBATCH --gres=gpu:2          # Request 2 GPUs
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl
#SBATCH --nodes=1             # 1 node
#SBATCH --cpus-per-task=4     # 8 CPUs per task
#SBATCH --mem=30GB            # 30GB of memory
#SBATCH --time=24:00:00        # runtime

cd /home/vsl333/frame-align

# # Name of the environment file
# ENV_FILE="env.yml"

# # Check if the file exists
# if [[ -f "$ENV_FILE" ]]; then
#     echo "Creating conda environment from $ENV_FILE..."
    
#     # Create the conda environment from the yml file
#     conda env create -f "$ENV_FILE"
    
#     # Check if the environment was created successfully
#     if [[ $? -eq 0 ]]; then
#         echo "Conda environment created successfully!"
#     else
#         echo "Failed to create conda environment."
#     fi
# else
#     echo "$ENV_FILE not found. Please ensure the file exists."
# fi


# # Activate the conda environment
# conda activate frame-env

source /home/vsl333/frame-align/frame-env/bin/activate
hostname
echo $CUDA_VISIBLE_DEVICES

python3 /home/vsl333/frame-align/src/models/vision/run_prompts.py
