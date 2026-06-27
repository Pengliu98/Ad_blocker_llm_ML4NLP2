import json
import torch
from torch.utils.data import Dataset


def load_sentence_pair_data(pairs_filepath, labels_filepath):
    """
    Load official sentence-pair data and merge with labels.
    label=0 -> not ad
    label=1 -> ad (start)
    label=2 -> ad (continuation)
    We treat label 1 and 2 both as ad (binary: 0=not-ad, 1=ad)
    """
    labels_dict = {}
    with open(labels_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_label = item.get('label', 0)
            # NaN comes through as float in some rows — treat as 0
            if not isinstance(raw_label, int):
                raw_label = 0
            # Collapse label=2 into label=1 (both are ad sentences)
            binary_label = 1 if raw_label in (1, 2) else 0
            labels_dict[item['id']] = binary_label

    data = []
    with open(pairs_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pair_id    = item['id']
            sentence1  = item.get('sentence1', '') or ''
            sentence2  = item.get('sentence2', '') or ''
            if not sentence2.strip():
                continue
            label = labels_dict.get(pair_id, 0)
            data.append({
                'id':        pair_id,
                'sentence1': sentence1.strip(),
                'sentence2': sentence2.strip(),
                'label':     label,
            })

    ad_count     = sum(1 for d in data if d['label'] == 1)
    non_ad_count = sum(1 for d in data if d['label'] == 0)
    print(f'Loaded {len(data)} sentence pairs: '
          f'{ad_count} ad ({100*ad_count/len(data):.1f}%), '
          f'{non_ad_count} non-ad ({100*non_ad_count/len(data):.1f}%)')
    return data


def sentence_collate_fn(batch):
    input_ids      = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels         = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {
        'input_ids':      input_ids,
        'attention_mask': attention_mask,
        'labels':         labels,
    }


class ToucheSentencePairDataset(Dataset):
    def __init__(self, pairs_filepath, labels_filepath, tokenizer, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = load_sentence_pair_data(pairs_filepath, labels_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Encode sentence pair: [CLS] sentence1 [SEP] sentence2 [SEP]
        # This gives the model context (sentence1) + candidate (sentence2)
        encoding = self.tokenizer(
            item['sentence1'],
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          item['label'],
            'id':             item['id'],
        }
