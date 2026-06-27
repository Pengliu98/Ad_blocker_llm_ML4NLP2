import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import json
import argparse
from transformers import AutoTokenizer
from model_sentence import SentenceAdClassifier


def split_into_sentences(text):
    """
    Split document text into (sentence, start, end) tuples.
    Mirrors the official sentence-pairs format sliding window.
    """
    sentence_endings = {'.', '!', '?'}
    sentences = []
    current_start = 0
    i = 0

    while i < len(text):
        if text[i] in sentence_endings:
            next_i = i + 1
            while next_i < len(text) and text[next_i] == ' ':
                next_i += 1
            if next_i >= len(text) or text[next_i].isupper():
                sent = text[current_start:i + 1].strip()
                if sent:
                    sentences.append((sent, current_start, i + 1))
                current_start = next_i
                i = next_i
                continue
        i += 1

    remainder = text[current_start:].strip()
    if remainder:
        sentences.append((remainder, current_start, len(text)))

    return sentences


def predict_ad_spans(text, model, tokenizer, device, threshold=0.5, max_length=256):
    """
    Split document into sentences, classify each as ad/not-ad using
    the previous sentence as context (matching official sentence-pair format).
    Returns list of [start, end] character spans for ad sentences.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Build sentence pairs: (prev_sentence, current_sentence)
    # For the first sentence, use empty string as context
    sentence1_list = [''] + [s[0] for s in sentences[:-1]]
    sentence2_list = [s[0] for s in sentences]

    # Batch encode all pairs
    encodings = tokenizer(
        sentence1_list,
        sentence2_list,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )

    input_ids      = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu()

    ad_spans = []
    for i, (sent_text, sent_start, sent_end) in enumerate(sentences):
        if probs[i].item() >= threshold:
            ad_spans.append([sent_start, sent_end])

    return ad_spans


def load_threshold(model_dir, cli_threshold):
    if cli_threshold is not None:
        return cli_threshold
    thresh_path = os.path.join(model_dir, 'best_threshold.txt')
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            val = float(f.read().strip())
        print(f'Loaded threshold {val} from {thresh_path}')
        return val
    print('No threshold file found — using default 0.5')
    return 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   required=True)
    parser.add_argument('--output',    required=True)
    parser.add_argument('--task',      required=True, choices=['1', '2'])
    parser.add_argument('--model_dir', default='./saved_model_sentence')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--use_best',  action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    print('Loading model...')
    model = SentenceAdClassifier()
    weights_file = 'model_weights_best.pth' if args.use_best else 'model_weights.pth'
    weights_path = os.path.join(args.model_dir, weights_file)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Loaded weights from {weights_path}')

    threshold = load_threshold(args.model_dir, args.threshold)
    print(f'Using threshold: {threshold}')

    TEAM_TAG = 'ModernBERT-AdHunter-Sentence'

    print(f'Running inference for Task {args.task}...')
    with open(args.dataset, 'r', encoding='utf-8') as f_in, \
         open(args.output,  'w', encoding='utf-8') as f_out:

        for line in f_in:
            data   = json.loads(line)
            doc_id = data.get('id', '')
            text   = data.get('response', '')

            spans = []
            if text:
                spans = predict_ad_spans(
                    text, model, tokenizer, device, threshold
                )

            label = 1 if spans else 0

            if args.task == '1':
                out = {'id': doc_id, 'label': label, 'tag': TEAM_TAG}
            else:
                extracted = ' '.join(text[s:e] for s, e in spans)
                out = {
                    'id':       doc_id,
                    'response': extracted,
                    'spans':    spans,
                    'tag':      TEAM_TAG,
                }

            f_out.write(json.dumps(out) + '\n')

    print('Inference complete!')
