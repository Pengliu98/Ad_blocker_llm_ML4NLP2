import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

# Import your custom architecture and data loader
from model import ModernBertMultiTask
from data_loader import NativeAdDataset, custom_collate_fn

# 1. Setup Device & Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading saved model for Token-Level (Sklearn) Evaluation...")

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
true_tokens_flat = []
pred_tokens_flat = []

id_to_tag = {0: "O", 1: "B-AD", 2: "I-AD"}

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['bio_tags'].to(device)

        # Get token predictions
        _, token_logits = model(inputs, attention_mask)
        preds = torch.argmax(token_logits, dim=2)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        # FLATTEN THE LISTS FOR SKLEARN
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] != -100: # Ignore padding
                    true_tokens_flat.append(id_to_tag[labels[i][j]])
                    pred_tokens_flat.append(id_to_tag[preds[i][j]])

# 4. Print the Sklearn Scorecard
print("\n" + "="*50)
print("TOKEN-LEVEL EVALUATION REPORT (Sklearn - Partial Credit Allowed)")
print("="*50)
print(classification_report(true_tokens_flat, pred_tokens_flat))