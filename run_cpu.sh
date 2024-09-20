#!/bin/bash
#SBATCH --job-name=frame
#SBATCH --cpus-per-task=4 --mem=12G
#SBATCH --time=72:00:00

python src/models/text/extract_meta_topics.py