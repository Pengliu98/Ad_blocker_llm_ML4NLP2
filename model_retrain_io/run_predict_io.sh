#!/bin/bash
#SBATCH --job-name=PredictIO
#SBATCH --output=logs/predict_io_%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=teaching
#SBATCH --gpus=V100:1
#SBATCH --mem=16G

mkdir -p logs
module load apptainer

# 1. Force install packages
apptainer exec --nv conda.sif pip install --target . protobuf sentencepiece tiktoken huggingface-hub pandas

# 2. Run Task 1
apptainer exec --nv conda.sif python predict_io.py \
    --dataset data/responses-test.jsonl \
    --output my_tira_task1_io.jsonl \
    --task 1 \
    --threshold 0.55

# 3. Run Task 2
apptainer exec --nv conda.sif python predict_io.py \
    --dataset data/responses-test.jsonl \
    --output my_tira_task2_io.jsonl \
    --task 2 \
    --threshold 0.55

# 4. Prepare folders (remove old dirs first to prevent mixing predictions)
echo "Setting up evaluation directories..."
rm -rf my_predictions true_labels evaluation_results
mkdir -p my_predictions true_labels evaluation_results

cp my_tira_task2_io.jsonl my_predictions/
cp data/responses-test-labels.jsonl true_labels/

# 5. Run the official evaluator
echo "Running official evaluator..."
apptainer exec conda.sif python evaluator.py \
    -p my_predictions/ \
    -t true_labels/ \
    -o evaluation_results/ \
    --index-field id

echo "=================================================="
echo "          OFFICIAL EVALUATION RESULTS             "
echo "=================================================="
cat evaluation_results/*