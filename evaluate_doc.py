import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

# Import your custom architecture and data loader
from model import ModernBertMultiTask
from data_loader import NativeAdDataset, custom_collate_fn

# 1. Setup Device & Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading saved model for Document-Level Evaluation...")

tokenizer = AutoTokenizer.from_pretrained("./saved_model")
model = ModernBertMultiTask("answerdotai/ModernBERT-base")
model.load_state_dict(torch.load("./saved_model/model_weights.pth", map_location=device))
model.to(device)
model.eval()

# 2. Load the Evaluation Dataset
# ---> USE YOUR SAME VAL/TEST FILES HERE <---
eval_dataset = NativeAdDataset("data/responses-validation.jsonl", "data/responses-validation-labels.jsonl")

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    collate_fn=lambda b: custom_collate_fn(b, tokenizer)
)

# 3. Run Inference
true_doc_labels = []
pred_doc_labels = []

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Grab the Document-level labels (0 or 1)
        labels = batch['labels'].to(device) 

        # We only care about doc_logits this time!
        doc_logits, _ = model(inputs, attention_mask)
        
        # Get the Yes/No prediction
        preds = torch.argmax(doc_logits, dim=1)

        # Store results
        true_doc_labels.extend(labels.cpu().numpy())
        pred_doc_labels.extend(preds.cpu().numpy())

# 4. Print the Official Scorecard
print("\n" + "="*50)
print("DOCUMENT-LEVEL EVALUATION REPORT (Ad vs Non-Ad)")
print("="*50)
# '0' is Normal Text, '1' is Contains Ad
print(classification_report(true_doc_labels, pred_doc_labels, target_names=["Normal Text", "Contains Ad"]))