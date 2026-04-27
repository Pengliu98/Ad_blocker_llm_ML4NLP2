import torch
import torch.nn as nn
from transformers import AutoModel

class ModernBertMultiTask(nn.Module):
    def __init__(self, model_name = "answerdotai/ModernBERT-base"):
        super().__init__()

        # Load the shared encoder (ModernBERT)
        self.shared_encoder = AutoModel.from_pretrained(model_name)

        # Document-level classifier for binary classification (ad vs non-ad)
        self.doc_classifier = nn.Linear(in_features=768, out_features=2)

        # Token-level classifier for span detection (3 classes: non-ad, ad-start, ad-inside)
        self.token_classifier = nn.Linear(in_features=768, out_features=3)

        
    def forward(self, input_ids, attention_mask):

        # Pass the input through the shared encoder to obtain contextualized token representations
        output = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the sequence output from the encoder, which contains the contextualized representations for each token in the input sequence
        sequence_output = output.last_hidden_state

        # Use the [CLS] token representation for document-level classification （[batch_size, sequence_length, hidden_size]）
        cls_output = sequence_output[:, 0, :]

        # Pass the [CLS] token representation through the document-level classifier to get logits for ad vs non-ad classification
        doc_logits = self.doc_classifier(cls_output)

        # Pass the token representations through the token-level classifier to get logits for span detection (non-ad, ad-start, ad-inside)
        token_logits = self.token_classifier(sequence_output)

        return doc_logits, token_logits




