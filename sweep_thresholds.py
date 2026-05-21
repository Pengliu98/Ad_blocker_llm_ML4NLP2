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

eval_dataset = NativeAdDataset("data/responses-validation.jsonl", "data/responses-validation-labels.jsonl")

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    collate_fn=lambda b: custom_collate_fn(b, tokenizer)
)

thresholds_to_test = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

print("Starting Threshold Sweep...\n")

with torch.no_grad():
    # 1. Run the model ONCE and save the raw logits for the whole validation set
    all_logits = []
    all_labels = []
    
    for batch in eval_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['bio_tags'].to(device)
        
        _, token_logits = model(inputs, attention_mask)
        
        all_logits.append(token_logits)
        all_labels.append(labels)

# 2. Now loop through the different math rules (thresholds) without running the model again!
for threshold in thresholds_to_test:
    true_tokens_flat = []
    pred_tokens_flat = []
    
    for batch_idx in range(len(all_logits)):
        token_logits = all_logits[batch_idx]
        labels = all_labels[batch_idx].cpu().numpy()
        
        # Convert raw logits to percentages (Softmax)
        probs = torch.softmax(token_logits, dim=2).cpu().numpy()
        
        # Apply our custom threshold rule
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] != -100: # Ignore padding
                    true_tokens_flat.append(labels[i][j])
                    
                    token_probs = probs[i][j]
                    
                    # If the probability of being "Normal" (index 0) is less than (1 - threshold)
                    # Example: If threshold is 0.15 (15%), we trigger if Normal is < 85%
                    if token_probs[0] < (1.0 - threshold): 
                        if token_probs[1] > token_probs[2]:
                            pred_tokens_flat.append(1) # Predict B-AD
                        else:
                            pred_tokens_flat.append(2) # Predict I-AD
                    else:
                        pred_tokens_flat.append(0) # Predict Normal

    # 3. Print the results for this specific threshold
    print(f"=== Results for Confidence Threshold: {threshold * 100}% ===")
    print(classification_report(true_tokens_flat, pred_tokens_flat, target_names=["O", "B-AD", "I-AD"], zero_division=0))
    print("\n")