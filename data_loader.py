import json
from torch.utils import data
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

# Custom collate function to handle variable-length sequences and prepare batch data
def custom_collate_fn(batch, tokenizer):
        
        # Extract queries, responses, labels, and spans from the batch
        queries = []
        responses = []
        labels = []
        spans = []

        # Iterate through each item in the batch and collect the respective fields
        for item in batch:
            queries.append(item["query"])
            responses.append(item["response"])
            labels.append(item["label"])
            spans.append(item["spans"]) 

        # Tokenize the queries and responses using the provided tokenizer, ensuring proper padding and truncation
        tokenized_inputs = tokenizer(
            queries,
            responses,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # BIO tags for all 8 examples in the batch (for token-level classification)
        batch_bio_tags = []

        for i in range(len(batch)):
            offsets = tokenized_inputs["offset_mapping"][i]
            doc_span = spans[i]
            doc_tags = []

            if isinstance(doc_span, float):
                doc_span = []

            for offset in offsets:
                token_start, token_end = offset

                if token_start == 0 and token_end == 0:
                    doc_tags.append(-100) # Special token, ignore in loss calculation
                    continue

                tag = 0 # Default tag is "0" ("0" = outside)
               
                for span in doc_span:
                    span_start, span_end = span

                    if token_start >= span_start and token_end <= span_end:
                        if token_start == span_start:
                            tag = 1 # Beginning of an ad span
                        else:
                            tag = 2 # Inside an ad span
                        break

                doc_tags.append(tag)

            batch_bio_tags.append(doc_tags)

        # 1. Convert each individual list of tags into a tensor
        bio_tensors = [torch.tensor(tags, dtype=torch.long) for tags in batch_bio_tags]

        # 2. Pad them all to the length of the longest sentence, using -100 as the filler
        batch_bio_tags_tensor = pad_sequence(bio_tensors, batch_first=True, padding_value=-100)

        # Convert labels to a tensor format suitable for model training and evaluation
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Return a dictionary containing the tokenized inputs, attention masks, offset mappings, labels, and spans for the batch
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "offset_mapping": tokenized_inputs["offset_mapping"],
            "labels": labels_tensor,
            "spans": spans,
            "bio_tags": batch_bio_tags_tensor
        }

# Function to load data from response and label files, organizing it into a structured format for easy access
def load_data(response_file_path, response_labels_file_path):
        # Initialize an empty dictionary to store the loaded data and a list to keep track of the IDs
        data = {}
        id_list = []

        # Open the response and label files, read their contents line by line, and populate the data dictionary with the relevant information for each ID
        with open(response_file_path, "r") as f:
            with open(response_labels_file_path, "r") as f_labels:
                for line in f:
                    response = json.loads(line)
                    data[response["id"]] = {"response": response["response"], "query": response["query"]}
                    id_list.append(response["id"])
                
                for line in f_labels:
                    label = json.loads(line)
                    if label["id"] in data:
                        data[label["id"]]["label"] = label.get("label", None)
                        data[label["id"]]["spans"] = label.get("spans", [])
        
        return data, id_list
    
class NativeAdDataset(Dataset):

    def __init__(self, response_file_path, response_labels_file_path):
        self.data, self.id_list = load_data(response_file_path, response_labels_file_path)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        item_id = self.id_list[idx]
        item_data = self.data[item_id]
        return {
            "id": item_id,
            "query": item_data["query"],
            "response": item_data["response"],
            "label": item_data.get("label", None),
            "spans": item_data.get("spans", [])
        }


    
    
    