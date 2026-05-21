import json
import torch
import argparse
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from model import ModernBertMultiTask

def extract_ad_spans(text, model, tokenizer, device):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].numpy()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _, token_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.softmax(token_logits, dim=2)[0].cpu().numpy()
        
        preds = []
        for token_probs in probs:
            if token_probs[0] < 0.70: 
                preds.append(1) # Ad
            else:
                preds.append(0) # Normal
        
    spans = []
    current_span_start = None
    
    for idx, pred in enumerate(preds):
        start_char, end_char = offset_mapping[idx]
        if start_char == 0 and end_char == 0: 
            continue
            
        if pred == 1: 
            if current_span_start is None:
                current_span_start = int(start_char) # Start new span
            current_span_end = int(end_char)         # Keep pushing the end forward
        elif pred == 0: 
            if current_span_start is not None:
                spans.append([current_span_start, current_span_end]) 
                current_span_start = None
                
    if current_span_start is not None:
        spans.append([current_span_start, current_span_end])
        
    return spans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl")
    parser.add_argument("--task", type=str, required=True, choices=["1", "2"], help="Sub-task number (1 or 2)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the tokenizer from your repo
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # 2. Initialize your custom architecture
    print("Initializing multi-task architecture...")
    model = ModernBertMultiTask("answerdotai/ModernBERT-base")
    
    # 3. Download the fine-tuned weights directly from Hugging Face
    print("Downloading weights from Hugging Face...")
    weights_path = hf_hub_download(repo_id="Penggggg98/touche2026-adhunter", filename="saved_model_IO/model_weights.pth")
    
    # 4. Load the weights into the model
    print("Loading weights into model...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    TEAM_TAG = "ModernBERT-AdHunter"

    print(f"Starting inference for Task {args.task}...")
    with open(args.dataset, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            doc_id = data.get("id", "")
            raw_text = data.get("response", "")
            
            spans = []
            if raw_text:
                spans = extract_ad_spans(raw_text, model, tokenizer, device)
            
            # Sub-Task 1 Logic: If there are spans, label is 1 (Ad). Otherwise 0.
            label = 1 if len(spans) > 0 else 0

            # Output ONLY the strict keys required by the specific Sub-Task
            if args.task == "1":
                tira_output = {
                    "id": doc_id,
                    "label": label,
                    "tag": TEAM_TAG
                }
            elif args.task == "2":
                # Extract the exact advertisement text based on the generated spans
                extracted_texts = [raw_text[start:end] for start, end in spans]
                response_text = " ".join(extracted_texts)

                tira_output = {
                    "id": doc_id,
                    "response": response_text,
                    "spans": spans,
                    "tag": TEAM_TAG
                }
            
            f_out.write(json.dumps(tira_output) + '\n')
            
    print("Inference complete!")