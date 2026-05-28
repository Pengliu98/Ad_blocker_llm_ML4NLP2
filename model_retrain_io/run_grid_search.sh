#!/bin/bash
#SBATCH --job-name=GridSearch
#SBATCH --output=logs/grid_search_%j.out
#SBATCH --time=04:00:00            
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1              
#SBATCH --mem=32G                  

mkdir -p logs
module load apptainer

# 1. Ensure required packages are present in the container
apptainer exec --nv conda.sif pip install --target . protobuf sentencepiece tiktoken huggingface-hub

# 2. Run the hyperparameter grid search
echo "Starting grid search..."
apptainer exec --nv conda.sif python grid_search.py
echo "Grid search complete!"