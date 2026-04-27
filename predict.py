import json
import torch
from transformers import AutoTokenizer
from model import ModernBertMultiTask

# 1. Setup Device & Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("Loading the weighted model...")

# Point to your NEW weighted model folder
model_path = "./saved_model_weighted"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ModernBertMultiTask("answerdotai/ModernBERT-base")
model.load_state_dict(torch.load(f"{model_path}/model_weights.pth", map_location=device))
model.to(device)
model.eval() # Test mode!

# 2. The Extractor Function
def extract_ad_spans(text, model, tokenizer, device):
    # Tokenize with offset_mapping to get character boundaries
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].numpy()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # with torch.no_grad():
    #     _, token_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    #     preds = torch.argmax(token_logits, dim=2)[0].cpu().numpy()

    with torch.no_grad():
        _, token_logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        
        # Get the raw probabilities instead of just the highest guess
        probs = torch.softmax(token_logits, dim=2)[0].cpu().numpy()
        
        preds = []
        for token_probs in probs:
            # token_probs looks like [prob_O, prob_B_AD, prob_I_AD]
            # If the model is less than 80% sure this is Normal text... 
            # (Meaning it has at least a 20% suspicion it's an ad)
            if token_probs[0] < 0.80: 
                # Force it to pick the highest ad tag
                if token_probs[1] > token_probs[2]:
                    preds.append(1) # B-AD
                else:
                    preds.append(2) # I-AD
            else:
                preds.append(0) # O
        
    spans = []
    current_span = None
    
    # Translate tags back to character indices
    for idx, pred in enumerate(preds):
        start_char, end_char = offset_mapping[idx]
        if start_char == 0 and end_char == 0: # Skip special tokens like [CLS]
            continue
            
        tag = pred # 0: O, 1: B-AD, 2: I-AD
        
        if tag == 1: # B-AD
            if current_span: spans.append(current_span)
            current_span = {"start": int(start_char), "end": int(end_char)}
        elif tag == 2: # I-AD
            if current_span:
                current_span["end"] = int(end_char)
            else:
                current_span = {"start": int(start_char), "end": int(end_char)}
        elif tag == 0: # O
            if current_span:
                spans.append(current_span)
                current_span = None
                
    if current_span:
        spans.append(current_span)
        
    return spans

# 3. Process the Data
# ---> UPDATE THESE VARIABLES! <---
input_file = "data/responses-validation.jsonl" 
output_file = "final_predictions.jsonl"
text_json_key = "response" 

print(f"Extracting spans for {input_file}...")

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        raw_text = data.get(text_json_key, "")
        
        if raw_text:
            # Get the exact character boundaries
            predicted_spans = extract_ad_spans(raw_text, model, tokenizer, device)
            # Add them to the json object
            data['predicted_ad_spans'] = predicted_spans
        else:
            data['predicted_ad_spans'] = []
            
        # Write the line to your new submission file
        f_out.write(json.dumps(data) + '\n')

print(f"SUCCESS! All answers saved to {output_file}")