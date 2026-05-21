import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

from model import ModernBertMultiTask
from data_loader import NativeAdDataset, custom_collate_fn

# 1. Setup Device & Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading the IO model for Threshold Sweep...")

tokenizer = AutoTokenizer.from_pretrained("./saved_model_IO")
model = ModernBertMultiTask("answerdotai/ModernBERT-base")
model.load_state_dict(torch.load("./saved_model_IO/model_weights.pth", map_location=device))
model.to(device)
model.eval()

# 2. Load the Evaluation Dataset
eval_dataset = NativeAdDataset("data/responses-validation.jsonl", "data/responses-validation-labels.jsonl")
eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=lambda b: custom_collate_fn(b, tokenizer))

# Testing a wider range of confidence thresholds
thresholds_to_test = [0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90]

print("Starting Threshold Sweep...\n")

with torch.no_grad():
    all_logits = []
    all_labels = []
    
    for batch in eval_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['bio_tags'].to(device)
        
        _, token_logits = model(inputs, attention_mask)
        
        all_logits.append(token_logits)
        all_labels.append(labels)

# 3. Sweep through thresholds
for threshold in thresholds_to_test:
    true_tokens_flat = []
    pred_tokens_flat = []
    
    for batch_idx in range(len(all_logits)):
        token_logits = all_logits[batch_idx]
        labels = all_labels[batch_idx].cpu().numpy()
        
        probs = torch.softmax(token_logits, dim=2).cpu().numpy()
        
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] != -100: 
                    true_tokens_flat.append(labels[i][j])
                    
                    # token_probs[1] is the model's confidence that it is an Ad
                    ad_confidence = probs[i][j][1]
                    
                    if ad_confidence >= threshold: 
                        pred_tokens_flat.append(1) # Predict Ad
                    else:
                        pred_tokens_flat.append(0) # Predict Normal

    print(f"=== Results for Ad Confidence Threshold: {threshold * 100}% ===")
    print(classification_report(true_tokens_flat, pred_tokens_flat, target_names=["Normal", "Ad"], zero_division=0))
    print("\n")