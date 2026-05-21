import json

predictions_file = "my_tira_task1.jsonl" 
true_labels_file = "data/responses-test-labels.jsonl"

print("Loading files for Document-Level Classification Evaluation (Sub-Task 1)...")

# 1. Load Predictions
preds_dict = {}
with open(predictions_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        preds_dict[data['id']] = data['label']

# 2. Load True Labels
true_dict = {}
with open(true_labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # Some ground truth files have a "label" key. 
        # If not, we just check if the "spans" list is empty!
        if 'label' in data:
            true_label = data['label']
        else:
            true_label = 1 if len(data.get('spans', [])) > 0 else 0
            
        true_dict[data['id']] = true_label

# 3. Calculate Confusion Matrix Metrics
true_positives = 0  # Model guessed Ad (1), True was Ad (1)
false_positives = 0 # Model guessed Ad (1), True was Normal (0)
false_negatives = 0 # Model guessed Normal (0), True was Ad (1)
true_negatives = 0  # Model guessed Normal (0), True was Normal (0)

for doc_id in true_dict.keys():
    true_lbl = true_dict.get(doc_id, 0)
    pred_lbl = preds_dict.get(doc_id, 0) # Default to 0 if missing
    
    if true_lbl == 1 and pred_lbl == 1:
        true_positives += 1
    elif true_lbl == 0 and pred_lbl == 1:
        false_positives += 1
    elif true_lbl == 1 and pred_lbl == 0:
        false_negatives += 1
    elif true_lbl == 0 and pred_lbl == 0:
        true_negatives += 1

# 4. Calculate Final Metrics
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = (true_positives + true_negatives) / len(true_dict) if len(true_dict) > 0 else 0.0

print("\n" + "="*50)
print("FINAL EVALUATION (Sub-Task 1: Document Classification)")
print("="*50)
print(f"Total Documents Evaluated: {len(true_dict)}")
print(f"True Positives (Correct Ads):     {true_positives}")
print(f"False Positives (False Alarms):   {false_positives}")
print(f"False Negatives (Missed Ads):     {false_negatives}")
print(f"True Negatives (Correct Normal):  {true_negatives}")
print("-" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
print("="*50)