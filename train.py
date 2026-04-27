import torch
import torch.nn as nn
import model
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from data_loader import NativeAdDataset, custom_collate_fn
from model import ModernBertMultiTask

from torch.optim import AdamW



tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# Load the train dataset
response_file_path = "data/responses-train.jsonl"
response_labels_file_path = "data/responses-train-labels.jsonl"

# load the validation dataset
val_response_file_path = "data/responses-validation.jsonl"
val_response_labels_file_path = "data/responses-validation-labels.jsonl"

# Create an instance of the NativeAdDataset using the provided file paths
train_dataset = NativeAdDataset(response_file_path, response_labels_file_path)

# Create a DataLoader for the training dataset, specifying the batch size and the custom collate function for batching
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=lambda b: custom_collate_fn(b, tokenizer))

# Create an instance of the NativeAdDataset for the validation dataset
val_dataset = NativeAdDataset(val_response_file_path, val_response_labels_file_path)

# Create a DataLoader for the validation dataset, specifying the batch size and the custom collate function for batching
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=lambda b: custom_collate_fn(b, tokenizer))



# Initialize the ModernBertMultiTask model, which will be used for training on the native ad detection task
model = ModernBertMultiTask()

# Initialize the model and move it to the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Define the optimizer (AdamW) for training the model, specifying the learning rate and weight decay for regularization
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Training loop for the model, iterating over the specified number of epochs

num_epochs = 3

for epoch in range(num_epochs):

    model.train() # Set the model to training mode

    # Initialize variables to accumulate training loss and metrics for this epoch
    total_train_loss = 0.0

    for batch in train_dataloader:

        # Move the input data and labels to the appropriate device (GPU or CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bio_tags = batch["bio_tags"].to(device)

        # Zero the gradients of the optimizer to prepare for the backward pass
        optimizer.zero_grad()

        # Forward pass through the model to obtain document-level and token-level logits
        doc_logits, token_logits = model(input_ids = input_ids, attention_mask = attention_mask)

        # Compute the loss for document-level classification (binary cross-entropy loss)
        doc_loss_fn = nn.CrossEntropyLoss()
        doc_loss = doc_loss_fn(doc_logits, labels)

        # Compute the loss for token-level classification (cross-entropy loss for span detection)
        # CLASS WEIGHTS: Class 0 (O) = 1.0, Class 1 (B-AD) = 100.0, Class 2 (I-AD) = 1.0
        class_weights = torch.tensor([1.0, 100.0, 1.0], device=device)
        token_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        flat_token_logits = token_logits.view(-1,3) # Reshape token logits for loss computation
        flat_token_labels = bio_tags.view(-1) # Reshape token labels for
        token_loss = token_loss_fn(flat_token_logits, flat_token_labels)

        # Combine the document-level and token-level losses to get the total loss for backpropagation
        total_loss = doc_loss + token_loss

        # Accumulate the total training loss across batches for this epoch
        total_train_loss += total_loss.item()

        # Backward pass to compute gradients and update model parameters
        total_loss.backward()
        optimizer.step()

    # After completing the training loop for this epoch, compute the average training loss and print it for monitoring
    avg_train_loss = total_train_loss / len(train_dataloader)
    


    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculation for evaluation

        # Initialize variables to accumulate validation loss and metrics
        total_val_loss = 0.0

        

        for batch in val_dataloader:

            # Move the input data and labels to the appropriate device (GPU or CPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            bio_tags = batch["bio_tags"].to(device)

            # Forward pass through the model to obtain document-level and token-level logits
            doc_logits, token_logits = model(input_ids = input_ids, attention_mask = attention_mask)

            # Compute the loss for document-level classification (binary cross-entropy loss) on the validation set
            doc_loss_fn = nn.CrossEntropyLoss()
            doc_loss = doc_loss_fn(doc_logits, labels)
            
        

            # Compute the loss for token-level classification (cross-entropy loss for span detection) on the validation set
            class_weights = torch.tensor([1.0, 100.0, 1.0], device=device)
            token_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            flat_token_logits = token_logits.view(-1,3) # Reshape token logits for
            flat_token_labels = bio_tags.view(-1) # Reshape token labels for loss computation
            token_loss = token_loss_fn(flat_token_logits, flat_token_labels)

            # Combine the document-level and token-level losses to get the total validation loss for this batch
            batch_loss = doc_loss + token_loss
        
            
            # Accumulate the total validation loss across batches
            total_val_loss += batch_loss.item()

    # After completing the validation loop for this epoch, compute the average validation loss and print it for monitoring
    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch+1}/{num_epochs} - Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
    print("-" * 50) 

import os


# Create a NEW folder for version 2
os.makedirs("./saved_model_weighted", exist_ok=True)

# Save the native PyTorch model weights to the new folder
torch.save(model.state_dict(), "./saved_model_weighted/model_weights.pth")

# Save the Hugging Face tokenizer rules to the new folder
tokenizer.save_pretrained("./saved_model_weighted")

print("SUCCESS: Model weights and Tokenizer saved to ./saved_model_weighted!")



        









