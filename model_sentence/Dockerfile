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

# 5. PRE-CACHE the backbone
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base'); AutoModel.from_pretrained('answerdotai/ModernBERT-base')"

# 6. DOWNLOAD the entire Hugging Face repo directly into the folder
RUN huggingface-cli download Penggggg98/touche2026-adhunter-sentence --local-dir /saved_model_sentence

# 7. Set the python path to root so imports work
ENV PYTHONPATH=/

# 8. Set the entrypoint
ENTRYPOINT ["python3", "/predict.py", "--model_dir", "/saved_model_sentence"]
