import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch, torch._dynamo
torch._dynamo.config.suppress_errors = True

import json, torch
from transformers import AutoTokenizer
from model_io import ModernBertMultiTaskIO
from predict_io import extract_ad_spans, merge_spans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained("saved_model_IO")
model = ModernBertMultiTaskIO()
model.load_state_dict(torch.load("saved_model_IO/model_weights.pth", map_location=device))
model.to(device)
model.eval()

# Load gold spans
gold = {}
with open("data/responses-validation-labels.jsonl") as f:
    for line in f:
        item = json.loads(line)
        spans = item.get("sentence_spans") or item.get("spans")
        gold[item["id"]] = spans if isinstance(spans, list) else []

# Load docs
docs = []
with open("data/responses-validation.jsonl") as f:
    for line in f:
        item = json.loads(line)
        if item.get("response"):
            docs.append(item)

print(f"Evaluating on {len(docs)} docs...")

# Pre-compute per-character ad probabilities once (expensive part)
# then threshold sweep is just numpy ops
from predict_io import extract_ad_spans

results = []
for i, item in enumerate(docs):
    if i % 100 == 0:
        print(f"  {i}/{len(docs)}")
    doc_id = item["id"]
    text   = item["response"]
    gold_chars = set()
    for s, e in gold.get(doc_id, []):
        gold_chars.update(range(s, e))
    results.append((doc_id, text, gold_chars))

print("\nThreshold + gap sweep:")
print(f"{'thresh':>8} {'gap':>5} {'P':>7} {'R':>7} {'F1':>7}")
print("-" * 40)

best_f1 = 0
best_config = (0.5, 10)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    for gap in [10, 25, 50]:
        tp = fp = fn = 0
        for doc_id, text, gold_chars in results:
            pred_spans = extract_ad_spans(text, model, tokenizer, device, threshold)
            pred_spans = [[s, e] for s, e in pred_spans]  # already merged inside
            # re-merge with this gap
            pred_spans = merge_spans(pred_spans, gap=gap)
            pred_chars = set()
            for s, e in pred_spans:
                pred_chars.update(range(s, e))
            tp += len(pred_chars & gold_chars)
            fp += len(pred_chars - gold_chars)
            fn += len(gold_chars - pred_chars)

        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        marker = " ← best" if f1 > best_f1 else ""
        print(f"{threshold:>8.2f} {gap:>5d} {p:>7.3f} {r:>7.3f} {f1:>7.3f}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_config = (threshold, gap)

print(f"\nBest: threshold={best_config[0]}, gap={best_config[1]}, F1={best_f1:.3f}")
print(f"Update best_threshold.txt and re-run predict with --threshold {best_config[0]}")