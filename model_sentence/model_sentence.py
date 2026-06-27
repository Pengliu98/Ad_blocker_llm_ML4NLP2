import torch.nn as nn
from transformers import AutoModel


class SentenceAdClassifier(nn.Module):
    """
    Sentence-pair classifier: given (sentence1, sentence2), predict
    whether sentence2 is an ad sentence.

    Input: [CLS] sentence1 [SEP] sentence2 [SEP]
    The [CLS] token captures the relationship between both sentences,
    helping the model use context to detect subtle native ads.
    """

    def __init__(self, model_name='answerdotai/ModernBERT-base', dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # 0=not-ad, 1=ad

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = output.last_hidden_state[:, 0, :]    # (batch, hidden)
        logits = self.classifier(self.dropout(cls))   # (batch, 2)
        return logits
