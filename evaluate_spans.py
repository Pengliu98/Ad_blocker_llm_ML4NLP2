import json

predictions_file = "final_predictions.jsonl"
true_labels_file = "data/responses-validation-labels.jsonl"

print("Loading files for Character-Level Overlap Evaluation...")

# 1. Load Predictions
preds_dict = {}
with open(predictions_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # Convert spans to a set of individual character indices
        char_indices = set()
        for span in data.get('predicted_ad_spans', []):
            # Add every single character index (e.g., start 10 to end 15 -> {10, 11, 12, 13, 14})
            char_indices.update(range(span['start'], span['end']))
        preds_dict[data['id']] = char_indices

# 2. Load True Labels
true_dict = {}
with open(true_labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        true_spans_raw = data.get('spans', []) 
        if not isinstance(true_spans_raw, list):
            true_spans_raw = []
            
        char_indices = set()
        for span in true_spans_raw:
            if isinstance(span, list) and len(span) >= 2:
                char_indices.update(range(span[0], span[1]))
                
        true_dict[data['id']] = char_indices

# 3. Calculate Character-Level Math
total_true_chars = 0
total_pred_chars = 0
total_correct_chars = 0

for doc_id in true_dict.keys():
    true_chars = true_dict.get(doc_id, set())
    pred_chars = preds_dict.get(doc_id, set())
    
    total_true_chars += len(true_chars)
    total_pred_chars += len(pred_chars)
    total_correct_chars += len(true_chars.intersection(pred_chars))

# 4. Calculate Final Metrics
precision = total_correct_chars / total_pred_chars if total_pred_chars > 0 else 0.0
recall = total_correct_chars / total_true_chars if total_true_chars > 0 else 0.0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n" + "="*50)
print("FINAL EVALUATION (Character-Level Overlap)")
print("="*50)
print(f"Total True Ad Characters:       {total_true_chars}")
print(f"Total Predicted Ad Characters:  {total_pred_chars}")
print(f"Perfectly Matched Characters:   {total_correct_chars}")
print("-" * 50)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
print("="*50)