import torch.nn as nn
from transformers import AutoModel


class ModernBertMultiTaskIO(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.shared_encoder.config.hidden_size
        self.doc_classifier   = nn.Linear(hidden_size, 2)
        self.token_classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.shared_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        seq = output.last_hidden_state
        cls = seq[:, 0, :]
        doc_logits   = self.doc_classifier(cls)
        token_logits = self.token_classifier(seq)
        return doc_logits, token_logits