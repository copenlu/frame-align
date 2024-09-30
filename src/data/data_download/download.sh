#!/bin/bash
#SBATCH --job-name=sy_img   # Job name
#SBATCH --array=0-11          # Number of tasks
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl
#SBATCH --nodes=1             # 1 node
#SBATCH --cpus-per-task=4     # 8 CPUs per task
#SBATCH --mem=20GB            # 20GB of memory
#SBATCH --time=48:00:00        # runtime


# # Load any necessary modules (e.g., for Python)
# module load python/3.9.9

# # Create a virtual environment
# python3 -m venv my_env

cd /home/vsl333/frame-align

# Activate the virtual environment
source frame-cluster-env/bin/activate


# source my_env/bin/activate --> add this after installing dependencies locally
hostname
echo $CUDA_VISIBLE_DEVICES

base_dir="/projects/frame_align/data"

# Define directories (update these with actual paths)
image_dir="/projects/frame_align/data/news-img-data"

# Get the directory name for the current SLURM array task
directory_name=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" src/data/data_download/directories.txt)

# Print some debug info
# TO DO: Add "2023-06-01_2023-06-30" to directories.txt later
echo "Processing directory: $directory_name"
hostname
echo $CUDA_VISIBLE_DEVICES

# Run the Python script with the appropriate arguments
python3 src/data/data_download/download_img.py "$base_dir" "$directory_name" "$image_dir"