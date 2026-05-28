import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from transformers import AutoTokenizer

from dataset_io import ToucheIODataset, io_collate_fn
from model_io import ModernBertMultiTaskIO

# ─────────────────────────────────────────────
# CONFIG — flip FINAL_SUBMISSION to True once
# you have found your best threshold on val set
# ─────────────────────────────────────────────
FINAL_SUBMISSION = False   # True = merge val into train, skip threshold search
NUM_EPOCHS       = 8
BATCH_SIZE       = 8
LEARNING_RATE    = 2e-5
AD_OVERSAMPLE    = 5.0     # how much to up-weight ad documents in sampler
DOC_AD_WEIGHT    = 5.0     # class weight for ad documents in doc loss
TOKEN_AD_WEIGHT  = 10.0    # focal loss positive weight for ad tokens
FOCAL_GAMMA      = 2.0     # focal loss gamma — higher = more focus on hard examples
OUTPUT_DIR       = "./saved_model_IO"

TRAIN_RESPONSES = "data/responses-train.jsonl"
TRAIN_LABELS    = "data/responses-train-labels.jsonl"
VAL_RESPONSES   = "data/responses-validation.jsonl"
VAL_LABELS      = "data/responses-validation-labels.jsonl"


# ─────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=10.0, ignore_index=-100):
        super().__init__()
        self.gamma        = gamma
        self.pos_weight   = pos_weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        weight = torch.tensor([1.0, self.pos_weight], device=logits.device)
        ce  = F.cross_entropy(
            logits, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        mask = targets != self.ignore_index
        return loss[mask].mean()


# ─────────────────────────────────────────────
# Weighted sampler helper
# ─────────────────────────────────────────────
def make_sampler(dataset_items):
    weights = [AD_OVERSAMPLE if item["label"] == 1 else 1.0
               for item in dataset_items]
    return WeightedRandomSampler(weights, num_samples=len(weights))


# ─────────────────────────────────────────────
# Per-epoch threshold search on validation set
# ─────────────────────────────────────────────
def search_threshold(model, val_loader, device, epoch):
    model.eval()
    all_probs = []
    all_gold  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_labels   = batch["io_tags"]          # stay on CPU

            _, token_logits = model(input_ids, attention_mask)
            probs = torch.softmax(token_logits.cpu(), dim=-1)[:, :, 1]  # ad prob

            mask = token_labels != -100
            all_probs.append(probs[mask])
            all_gold.append(token_labels[mask])

    all_probs = torch.cat(all_probs)
    all_gold  = torch.cat(all_gold)

    print(f"\n  Threshold search (epoch {epoch}):")
    best_f1, best_thresh = 0.0, 0.5

    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = (all_probs >= thresh).long()
        tp = ((preds == 1) & (all_gold == 1)).sum().item()
        fp = ((preds == 1) & (all_gold == 0)).sum().item()
        fn = ((preds == 0) & (all_gold == 1)).sum().item()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        marker = "  ← best" if f1 > best_f1 else ""
        print(f"    thresh={thresh:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}{marker}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"  → Best: thresh={best_thresh}  F1={best_f1:.3f}\n")
    model.train()
    return best_thresh, best_f1


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Mode: {'FINAL SUBMISSION' if FINAL_SUBMISSION else 'DEVELOPMENT'}")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────
    train_dataset = ToucheIODataset(TRAIN_RESPONSES, TRAIN_LABELS, tokenizer)
    val_dataset   = ToucheIODataset(VAL_RESPONSES,   VAL_LABELS,   tokenizer)

    print(f"Train docs: {len(train_dataset)}  |  Val docs: {len(val_dataset)}")

    if FINAL_SUBMISSION:
        # Merge train + val for final model
        combined  = ConcatDataset([train_dataset, val_dataset])
        all_items = train_dataset.data + val_dataset.data
        sampler   = make_sampler(all_items)
        train_loader = DataLoader(
            combined, batch_size=BATCH_SIZE,
            sampler=sampler, collate_fn=io_collate_fn,
        )
        val_loader = None
        print(f"Combined dataset: {len(combined)} docs")
    else:
        # Keep val separate for threshold search
        sampler = make_sampler(train_dataset.data)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            sampler=sampler, collate_fn=io_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16,
            shuffle=False, collate_fn=io_collate_fn,
        )

    # ── Model & optimiser ─────────────────────
    model     = ModernBertMultiTaskIO()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    doc_loss_fct   = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, DOC_AD_WEIGHT]).to(device)
    )
    token_loss_fct = FocalLoss(
        gamma=FOCAL_GAMMA,
        pos_weight=TOKEN_AD_WEIGHT,
        ignore_index=-100,
    )

    # ── Training loop ─────────────────────────
    best_val_f1     = 0.0
    best_threshold  = 0.5

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_labels     = batch["labels"].to(device)
            token_labels   = batch["io_tags"].to(device)

            optimizer.zero_grad()
            doc_logits, token_logits = model(input_ids, attention_mask)

            doc_loss   = doc_loss_fct(doc_logits, doc_labels)
            token_loss = token_loss_fct(
                token_logits.view(-1, 2),
                token_labels.view(-1),
            )
            loss = doc_loss + token_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} "
                    f"(doc={doc_loss.item():.4f}, tok={token_loss.item():.4f})"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch} complete | Avg loss: {avg_loss:.4f} ---")

        # Threshold search on validation (development mode only)
        if val_loader is not None:
            thresh, f1 = search_threshold(model, val_loader, device, epoch)
            if f1 > best_val_f1:
                best_val_f1    = f1
                best_threshold = thresh
                # Save best checkpoint
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(OUTPUT_DIR, "model_weights_best.pth"),
                )
                print(f"  ✓ New best checkpoint saved (thresh={best_threshold}, F1={best_val_f1:.3f})")

    # ── Save final model ──────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_weights.pth"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    if not FINAL_SUBMISSION:
        # Write best threshold to a file so predict_io.py can read it
        with open(os.path.join(OUTPUT_DIR, "best_threshold.txt"), "w") as f:
            f.write(str(best_threshold))
        print(f"\nBest threshold found: {best_threshold}  (val F1={best_val_f1:.3f})")
        print("Re-run with FINAL_SUBMISSION=True to train on train+val combined.")
    else:
        print(f"\nFinal model saved to {OUTPUT_DIR}")
        print(f"Use --threshold {best_threshold} when running predict_io.py")


if __name__ == "__main__":
    main()
