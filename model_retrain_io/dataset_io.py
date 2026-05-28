import json
import torch
from torch.utils.data import Dataset


def io_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    io_tags = torch.stack([item["io_tags"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "io_tags": io_tags,
    }


def load_io_data(responses_filepath, labels_filepath):
    labels_dict = {}
    with open(labels_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                labels_dict[item["id"]] = item

    data = []
    with open(responses_filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                doc_id = item["id"]
                if doc_id in labels_dict:
                    lbl = labels_dict[doc_id]
                    # Guard against null/None/nan spans (common when data
                    # was exported from pandas — empty cells become float nan)
                    raw_spans = lbl.get("sentence_spans") or lbl.get("spans")
                    ad_spans  = raw_spans if isinstance(raw_spans, list) else []
                    data.append({
                        "text": item["response"],
                        "label": lbl.get("label", 0),
                        "ad_spans": ad_spans,
                    })
    return data


class ToucheIODataset(Dataset):
    def __init__(self, responses_filepath, labels_filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_io_data(responses_filepath, labels_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        ad_spans = item["ad_spans"]
        if not isinstance(ad_spans, list):
            ad_spans = []
        doc_label = item["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping").squeeze(0)

        # Build character-level ad mask
        text_len = len(text)
        char_mask = torch.zeros(text_len, dtype=torch.bool)
        for start, end in ad_spans:
            s = max(0, start)
            e = min(text_len, end)
            if s < e:
                char_mask[s:e] = True

        # Map each token to IO label via its character offsets
        io_tags = []
        for start, end in offset_mapping.tolist():
            start, end = int(start), int(end)
            if start == 0 and end == 0:
                # Special token ([CLS], [SEP], padding)
                io_tags.append(-100)
            elif end <= text_len and char_mask[start:end].any():
                io_tags.append(1)   # Ad token
            else:
                io_tags.append(0)   # Non-ad token

        item_tensors = {key: val.squeeze(0) for key, val in encoding.items()}
        item_tensors["label"] = doc_label
        item_tensors["io_tags"] = torch.tensor(io_tags, dtype=torch.long)

        return item_tensors