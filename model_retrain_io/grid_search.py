import json, torch
from transformers import AutoTokenizer
from model_io import ModernBertMultiTaskIO
from predict_io import extract_ad_spans, merge_spans

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("saved_model_IO")
model = ModernBertMultiTaskIO()
model.load_state_dict(torch.load("saved_model_IO/model_weights.pth", map_location=device))
model.eval()

# Load val labels for scoring
gold = {}
with open("data/responses-validation-labels.jsonl") as f:
    for line in f:
        item = json.loads(line)
        spans = item.get("sentence_spans") or item.get("spans")
        gold[item["id"]] = spans if isinstance(spans, list) else []

for threshold in [0.4, 0.5, 0.6, 0.7]:
    for gap in [10, 25, 50]:
        tp = fp = fn = 0
        with open("data/responses-validation.jsonl") as f:
            for line in f:
                item = json.loads(line)
                doc_id = item["id"]
                text = item.get("response", "")
                if not text:
                    continue
                pred_spans = extract_ad_spans(text, model, tokenizer, device, threshold)
                pred_spans = merge_spans(pred_spans, gap=gap)  # re-merge with new gap
                gold_spans = gold.get(doc_id, [])

                # Simple token-level overlap scoring
                pred_chars = set()
                for s, e in pred_spans:
                    pred_chars.update(range(s, e))
                gold_chars = set()
                for s, e in gold_spans:
                    gold_chars.update(range(s, e))

                tp += len(pred_chars & gold_chars)
                fp += len(pred_chars - gold_chars)
                fn += len(gold_chars - pred_chars)

        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        print(f"thresh={threshold}  gap={gap:2d}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")