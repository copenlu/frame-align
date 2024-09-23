#!/bin/bash
#SBATCH --job-name=v_frame   # Job name
#SBATCH --gres=gpu:a100:1          # Request 1 GPUs
#SBATCH --partition=copenlu    # Partition to submit to
#SBATCH --cpus-per-task=4     # 8 CPUs per task
#SBATCH --mem=20GB            # 30GB of memory
#SBATCH --time=24:00:00        # runtime

python src/models/vision/vision_framing.py --data_file /projects/copenlu/data/arnav/frame-align/raw/2023-24/topic_samples.csv
