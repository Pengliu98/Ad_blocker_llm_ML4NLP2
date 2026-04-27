import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import classification_report

# Import your custom architecture and data loader functions
from model import ModernBertMultiTask
from data_loader import NativeAdDataset, custom_collate_fn

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load the Tokenizer and Model
print("Loading saved model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

# Initialize the blank custom architecture
model = ModernBertMultiTask("answerdotai/ModernBERT-base")

# Load your perfectly trained weights into the architecture
model.load_state_dict(torch.load("./saved_model/model_weights.pth", map_location=device))
model.to(device)

# CRITICAL: Put model in test mode (turns off dropout, stops learning)
model.eval() 

# 3. Load the Evaluation Dataset
# ---> UPDATE THESE TWO PATHS TO YOUR VALIDATION/TEST FILES <---
eval_dataset = NativeAdDataset("data/responses-validation.jsonl", "data/responses-validation-labels.jsonl")

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    collate_fn=lambda b: custom_collate_fn(b, tokenizer)
)

# 4. Run Inference
print("Running evaluation...")
true_labels = []
predictions = []

# Map your 0, 1, 2 integer tags back to standard BIO strings for the grader
id_to_tag = {0: "O", 1: "B-AD", 2: "I-AD"}

# Tell PyTorch NOT to calculate gradients (saves massive memory and time)
with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['bio_tags'].to(device)

        # Get model predictions
        doc_logits, token_logits = model(inputs, attention_mask)

        # Grab the highest probability prediction for each token
        preds = torch.argmax(token_logits, dim=2)

        # Convert tensors to python lists to grade them
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        for i in range(len(labels)):
            doc_true = []
            doc_preds = []
            for j in range(len(labels[i])):
                # CRITICAL: Ignore the -100 padding and subword tokens!
                if labels[i][j] != -100: 
                    doc_true.append(id_to_tag[labels[i][j]])
                    doc_preds.append(id_to_tag[preds[i][j]])
            
            true_labels.append(doc_true)
            predictions.append(doc_preds)

# 5. Print the Final Scorecard
print("\n" + "="*50)
print("FINAL EVALUATION REPORT")
print("="*50)
print(classification_report(true_labels, predictions))