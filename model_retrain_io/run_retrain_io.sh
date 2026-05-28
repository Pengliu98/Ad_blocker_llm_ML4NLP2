#!/bin/bash
#SBATCH --job-name=TrainIO
#SBATCH --output=logs/train_io_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1
#SBATCH --mem=32G

mkdir -p logs
module load apptainer

# Install required packages inside the container
apptainer exec --nv conda.sif pip install --target . protobuf sentencepiece tiktoken huggingface-hub

# Run the IO training script
apptainer exec --nv conda.sif python train_io.py