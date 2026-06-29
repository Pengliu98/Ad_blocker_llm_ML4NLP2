import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import json
import argparse
from transformers import AutoTokenizer
from model_io import ModernBertMultiTaskIO


def merge_spans(spans, gap=0):
    """Merge spans that are within `gap` characters of each other."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def split_into_sentences(text, spans):
    """
    Split merged ad spans into sentence-level chunks to match the
    granularity expected by the official evaluator.
    A sentence boundary is: punctuation (.!?) followed by whitespace
    and an uppercase letter (or end of span).
    """
    sentence_endings = {'.', '!', '?'}
    final_spans = []

    for start, end in spans:
        current_start = start
        i = start
        while i < end:
            if text[i] in sentence_endings:
                next_i = i + 1
                while next_i < end and text[next_i] == ' ':
                    next_i += 1
                if next_i >= end or text[next_i].isupper():
                    final_spans.append([current_start, i + 1])
                    current_start = next_i
                    i = next_i
                    continue
            i += 1

        if current_start < end:
            final_spans.append([current_start, end])

    # Strip leading/trailing whitespace from each span
    cleaned = []
    for s, e in final_spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            cleaned.append([s, e])

    return cleaned


def extract_ad_spans(text, model, tokenizer, device, threshold=0.5):
    """
    Run sliding-window inference over `text`.
    Returns:
      1. sentence-level ad spans based on token ad-probability.
      2. the document-level ad probability from the doc classifier.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        stride=128,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    del inputs["overflow_to_sample_mapping"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # FIXED: Extract both document and token logits
        doc_logits, token_logits = model(**inputs)

    # --- Sub-Task 1: Document Probability ---
    doc_probs = torch.softmax(doc_logits.cpu(), dim=-1)
    doc_prob = doc_probs[:, 1].mean().item()  # Average over all sliding windows

    # --- Sub-Task 2: Token Spans ---
    text_len = len(text)
    prob_sum = torch.zeros(text_len, dtype=torch.float32)
    counts   = torch.zeros(text_len, dtype=torch.float32)

    probs = torch.softmax(token_logits.cpu(), dim=-1)   # (chunks, seq_len, 2)

    for chunk_idx in range(probs.shape[0]):
        for i, (start, end) in enumerate(offset_mapping[chunk_idx].tolist()):
            start, end = int(start), int(end)
            if start == 0 and end == 0:
                continue
            if end > text_len:
                continue
            prob_sum[start:end] += probs[chunk_idx, i, 1]
            counts[start:end]   += 1

    avg_probs = prob_sum / counts.clamp(min=1)

    # Collect contiguous character spans above threshold
    raw_spans = []
    current   = None
    for i in range(text_len):
        if avg_probs[i].item() >= threshold:
            if current is None:
                current = [i, i + 1]
            else:
                current[1] = i + 1
        else:
            if current is not None:
                raw_spans.append(current)
                current = None
    if current is not None:
        raw_spans.append(current)

    # FIXED: Merge first, THEN filter short spans (and lowered the filter to 10 chars)
    # This prevents fragmented predictions from being discarded before they can merge
    merged = merge_spans(raw_spans, gap=0)
    merged = [s for s in merged if s[1] - s[0] >= 10]

    return split_into_sentences(text, merged), doc_prob


def load_threshold(model_dir, cli_threshold):
    if cli_threshold is not None:
        return cli_threshold
    thresh_path = os.path.join(model_dir, "best_threshold.txt")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            val = float(f.read().strip())
        print(f"Loaded threshold {val} from {thresh_path}")
        return val
    print("No threshold file found — using default 0.5")
    return 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,  help="Path to input .jsonl")
    parser.add_argument("--output",     required=True,  help="Path to output .jsonl")
    parser.add_argument("--task",       required=True,  choices=["1", "2"])
    parser.add_argument("--model_dir",  default="./saved_model_IO",
                        help="Directory containing model_weights.pth and tokenizer")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Ad probability threshold (overrides best_threshold.txt)")
    parser.add_argument("--use_best",   action="store_true",
                        help="Load model_weights_best.pth instead of model_weights.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    # FIX 1: Load directly from the pre-cached base model
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    print("Loading model...")
    model = ModernBertMultiTaskIO()
    weights_file = "model_weights_best.pth" if args.use_best else "model_weights.pth"
    # FIX 2: Force it to look in /saved_model_IO so the workflow can't confuse it
    weights_path = os.path.join("/saved_model_IO", weights_file)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    threshold = load_threshold(args.model_dir, args.threshold)
    print(f"Using threshold: {threshold}")

    TEAM_TAG = "ModernBERT-AdHunter-IO"

    print(f"Running inference for Task {args.task}...")
    with open(args.dataset, "r", encoding="utf-8") as f_in, \
         open(args.output,  "w", encoding="utf-8") as f_out:

        for line in f_in:
            data   = json.loads(line)
            doc_id = data.get("id", "")
            text   = data.get("response", "")

            spans = []
            doc_prob = 0.0
            if text:
                spans, doc_prob = extract_ad_spans(text, model, tokenizer, device, threshold)

            # FIXED: Sub-Task 1 now relies strictly on the dedicated document head
            label = 1 if doc_prob >= 0.5 else 0

            if args.task == "1":
                out = {
                    "id":    doc_id,
                    "label": label,
                    "tag":   TEAM_TAG,
                }
            else:
                extracted = " ".join(text[s:e] for s, e in spans)
                out = {
                    "id":       doc_id,
                    "response": extracted,
                    "spans":    spans,
                    "tag":      TEAM_TAG,
                }

            f_out.write(json.dumps(out) + "\n")

    print("Inference complete!")