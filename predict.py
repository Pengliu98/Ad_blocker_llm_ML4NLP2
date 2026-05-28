import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import json
import argparse
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from model import ModernBertMultiTaskIO


def merge_spans(spans, gap=0):
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

    cleaned = []
    for s, e in final_spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            cleaned.append([s, e])

    return cleaned


def extract_ad_spans(text, model, tokenizer, device, threshold=0.45):
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
        _, token_logits = model(**inputs)

    text_len = len(text)
    prob_sum = torch.zeros(text_len, dtype=torch.float32)
    counts   = torch.zeros(text_len, dtype=torch.float32)
    probs = torch.softmax(token_logits.cpu(), dim=-1)

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

    # Filter out very short spurious spans
    raw_spans = [s for s in raw_spans if s[1] - s[0] >= 20]

    merged = merge_spans(raw_spans, gap=0)
    return split_into_sentences(text, merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output",  type=str, required=True)
    parser.add_argument("--task",    type=str, required=True, choices=["1", "2"])
    args = parser.parse_args()

    # Threshold hardcoded — TIRA does not use shell scripts
    THRESHOLD = 0.45

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    print("Initializing model...")
    model = ModernBertMultiTaskIO()

    print("Downloading weights from HuggingFace...")
    weights_path = hf_hub_download(
        repo_id="Penggggg98/touche2026-adhunter",
        filename="saved_model_IO/model_weights.pth"
    )

    print("Loading weights...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    TEAM_TAG = "ModernBERT-AdHunter-IO"

    print(f"Running inference for Task {args.task} with threshold={THRESHOLD}...")
    with open(args.dataset, "r", encoding="utf-8") as f_in, \
         open(args.output,  "w", encoding="utf-8") as f_out:

        for line in f_in:
            data   = json.loads(line)
            doc_id = data.get("id", "")
            text   = data.get("response", "")

            spans = []
            if text:
                spans = extract_ad_spans(text, model, tokenizer, device, THRESHOLD)

            label = 1 if spans else 0

            if args.task == "1":
                out = {"id": doc_id, "label": label, "tag": TEAM_TAG}
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