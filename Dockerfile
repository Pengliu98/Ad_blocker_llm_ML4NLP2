FROM webis/touche25-ad-detection:0.0.1

# 2. Install huggingface-hub inside the image
RUN pip install --no-cache-dir huggingface-hub


ADD predict.py model.py /

# 4. PRE-CACHE THE MODEL AND TOKENIZER (Crucial for offline TIRA execution)
# This downloads the weights and the base tokenizer DURING the docker build phase
RUN python3 -c '\
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"); \
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id="Penggggg98/touche2026-adhunter", filename="saved_model_IO/model_weights.pth");'

# 5. Set the python path to root so imports work seamlessly
ENV PYTHONPATH=/

# 6. Set the entrypoint to your predict script
ENTRYPOINT ["python3", "/predict.py"]