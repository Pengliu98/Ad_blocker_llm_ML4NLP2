#!/bin/bash
#SBATCH --job-name=Eval_Doc
#SBATCH --output=logs/eval_doc_%j.out
#SBATCH --time=00:30:00           
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1              
#SBATCH --mem=16G                 

module load apptainer

apptainer exec --nv conda.sif bash -c "pip install --user scikit-learn && python evaluate_doc.py"