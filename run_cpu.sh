#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=12G
#SBATCH --time=72:00:00

# python src/data/get_topic_labels.py --data_dir /projects/copenlu/data/arnav/frame-align/raw/2023-24
python src/data/data_ops.py