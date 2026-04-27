#!/bin/bash
#SBATCH --job-name=Eval_Ads
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=00:30:00           
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1              
#SBATCH --mem=16G                 

module load apptainer

# We use "pip install --user" to safely install seqeval inside your home directory 
# before running the evaluation script!
apptainer exec --nv conda.sif bash -c "pip install --user seqeval && python evaluate.py"