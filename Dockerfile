# 1. Use the organizers' official baseline image
FROM webis/touche25-ad-detection:0.0.1

# 2. Install huggingface-hub inside the image
RUN pip install --no-cache-dir huggingface-hub

# 3. Add YOUR exact script files to the root directory
ADD predict.py model.py /

# 4. PRE-CACHE YOUR MODEL, BACKBONE, AND TOKENIZER
# This bakes all files into the image so they are available offline
RUN python3 -c '\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"); \
AutoModel.from_pretrained("answerdotai/ModernBERT-base"); \
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id="Penggggg98/touche2026-adhunter", filename="saved_model_IO/model_weights.pth");'

# 5. Set the python path to root so your imports work seamlessly
ENV PYTHONPATH=/

# 6. Set the entrypoint to your predict script
ENTRYPOINT ["python3", "/predict.py"]