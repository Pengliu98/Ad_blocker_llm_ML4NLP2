#!/bin/bash
#SBATCH --job-name=SweepThresh
#SBATCH --output=logs/sweep_%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1
#SBATCH --mem=16G

mkdir -p logs
module load apptainer

apptainer exec --nv conda.sif pip install --target . protobuf sentencepiece tiktoken huggingface-hub

apptainer exec --nv conda.sif python sweep_threshold.py