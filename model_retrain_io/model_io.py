import torch.nn as nn
from transformers import AutoModel


class ModernBertMultiTaskIO(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.shared_encoder.config.hidden_size

        # Task 1: Document-level classifier (0=No ad, 1=Ad)
        self.doc_classifier = nn.Linear(hidden_size, 2)

        # Task 2: Token-level classifier (0=O, 1=Ad)
        self.token_classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.shared_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        seq = output.last_hidden_state            # (batch, seq_len, hidden)
        cls = seq[:, 0, :]                        # (batch, hidden)

        doc_logits   = self.doc_classifier(cls)   # (batch, 2)
        token_logits = self.token_classifier(seq) # (batch, seq_len, 2)

        return doc_logits, token_logits
