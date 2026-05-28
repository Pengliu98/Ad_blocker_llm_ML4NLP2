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

# 4. Add script files to the root directory
#    predict_docker.py is renamed to predict.py
#    model_docker.py is renamed to model.py
ADD predict_docker.py /predict.py
ADD model_docker.py /model.py

# 5. PRE-CACHE the backbone, tokenizer, and your new IO model weights
RUN python3 -c '\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"); \
AutoModel.from_pretrained("answerdotai/ModernBERT-base"); \
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id="Penggggg98/touche2026-adhunter", filename="saved_model_IO/model_weights.pth");'

# 6. Set the python path to root so imports work
ENV PYTHONPATH=/

# 7. Set the entrypoint
ENTRYPOINT ["python3", "/predict.py"]