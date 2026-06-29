# 1. Use the organizers' official baseline image
FROM webis/touche25-ad-detection:0.0.1

# 2. Fix missing C compiler required by PyTorch Inductor for ModernBERT
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Install huggingface-hub inside the image
RUN pip install --no-cache-dir huggingface-hub

# 4. Add your sentence-level script files to the root directory
ADD predict_sentence.py /predict.py
ADD model_sentence.py /model_sentence.py
ADD dataset_sentence.py /dataset_sentence.py

# 5. PRE-CACHE the backbone and download your specific model files from HF
# NOTE: Make sure the repo variable matches your exact Hugging Face username and model name!
RUN python3 -c '\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"); \
AutoModel.from_pretrained("answerdotai/ModernBERT-base"); \
from huggingface_hub import hf_hub_download; \
repo = "Penggggg98/touche2026-adhunter-sentence"; \
for f in ["model_weights.pth", "best_threshold.txt", "tokenizer_config.json", "tokenizer.json"]: \
    hf_hub_download(repo_id=repo, filename=f, local_dir="/saved_model_sentence");'

# 6. Set the python path to root so imports work
ENV PYTHONPATH=/

# 7. Set the entrypoint
# We pass --model_dir to point directly to where we downloaded the files in step 5.
ENTRYPOINT ["python3", "/predict.py", "--model_dir", "/saved_model_sentence"]