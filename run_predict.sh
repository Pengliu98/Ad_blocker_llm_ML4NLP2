#!/bin/bash
#SBATCH --job-name=PredictSpans
#SBATCH --output=logs/predict_%j.out
#SBATCH --time=00:30:00           
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1              
#SBATCH --mem=16G                 

module load apptainer
apptainer exec --nv conda.sif python predict.py