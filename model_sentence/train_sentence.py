import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset_sentence import ToucheSentencePairDataset, sentence_collate_fn
from model_sentence import SentenceAdClassifier

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FINAL_SUBMISSION = False   # True = merge val into train, skip threshold search
NUM_EPOCHS       = 5
BATCH_SIZE       = 32
LEARNING_RATE    = 2e-5
AD_OVERSAMPLE    = 5.0     # oversample ad sentence pairs
AD_CLASS_WEIGHT  = 5.0     # focal loss positive weight
FOCAL_GAMMA      = 2.0
MAX_LENGTH       = 256
OUTPUT_DIR       = './saved_model_sentence'

TRAIN_PAIRS      = 'data/sentence-pairs-train.jsonl'
TRAIN_LABELS     = 'data/sentence-pairs-train-labels.jsonl'
VAL_PAIRS        = 'data/sentence-pairs-validation.jsonl'
VAL_LABELS       = 'data/sentence-pairs-validation-labels.jsonl'


# ─────────────────────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=5.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        weight = torch.tensor([1.0, self.pos_weight], device=logits.device)
        ce     = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt     = torch.exp(-ce)
        loss   = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────────────────────────
def make_sampler(dataset_items):
    weights = [AD_OVERSAMPLE if item['label'] == 1 else 1.0
               for item in dataset_items]
    return WeightedRandomSampler(weights, num_samples=len(weights))


# ─────────────────────────────────────────────────────────────
# Threshold search on validation set
# ─────────────────────────────────────────────────────────────
def search_threshold(model, val_loader, device, epoch):
    model.eval()
    all_probs = []
    all_gold  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            logits = model(input_ids, attention_mask)
            probs  = torch.softmax(logits.cpu(), dim=-1)[:, 1]

            all_probs.append(probs)
            all_gold.append(labels)

    all_probs = torch.cat(all_probs)
    all_gold  = torch.cat(all_gold)

    print(f'\n  Threshold search (epoch {epoch}):')
    best_f1, best_thresh = 0.0, 0.5

    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        preds = (all_probs >= thresh).long()
        tp = ((preds == 1) & (all_gold == 1)).sum().item()
        fp = ((preds == 1) & (all_gold == 0)).sum().item()
        fn = ((preds == 0) & (all_gold == 1)).sum().item()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        marker = '  <- best' if f1 > best_f1 else ''
        print(f'    thresh={thresh:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}{marker}')
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f'  -> Best: thresh={best_thresh}  F1={best_f1:.3f}\n')
    model.train()
    return best_thresh, best_f1


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Mode:   {"FINAL SUBMISSION" if FINAL_SUBMISSION else "DEVELOPMENT"}')
    print(f'Device: {device}')

    # ── Datasets ──────────────────────────────────────────────
    print('\nLoading training data...')
    train_dataset = ToucheSentencePairDataset(
        TRAIN_PAIRS, TRAIN_LABELS, tokenizer, MAX_LENGTH
    )
    print('Loading validation data...')
    val_dataset = ToucheSentencePairDataset(
        VAL_PAIRS, VAL_LABELS, tokenizer, MAX_LENGTH
    )

    if FINAL_SUBMISSION:
        combined  = ConcatDataset([train_dataset, val_dataset])
        all_items = train_dataset.data + val_dataset.data
        sampler   = make_sampler(all_items)
        train_loader = DataLoader(
            combined, batch_size=BATCH_SIZE,
            sampler=sampler, collate_fn=sentence_collate_fn,
        )
        val_loader = None
        print(f'Combined: {len(combined)} sentence pairs')
    else:
        sampler = make_sampler(train_dataset.data)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            sampler=sampler, collate_fn=sentence_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64,
            shuffle=False, collate_fn=sentence_collate_fn,
        )

    # ── Model ─────────────────────────────────────────────────
    model     = SentenceAdClassifier()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fct = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=AD_CLASS_WEIGHT)

    # ── Training loop ─────────────────────────────────────────
    best_val_f1    = 0.0
    best_threshold = 0.5

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = loss_fct(logits, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f'Epoch {epoch}/{NUM_EPOCHS} | '
                      f'Step {step}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'--- Epoch {epoch} complete | Avg loss: {avg_loss:.4f} ---')

        # Save checkpoint every epoch
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(OUTPUT_DIR, f'model_weights_epoch{epoch}.pth')
        )
        print(f'Checkpoint saved: model_weights_epoch{epoch}.pth')

        # Threshold search on validation
        if val_loader is not None:
            thresh, f1 = search_threshold(model, val_loader, device, epoch)
            if f1 > best_val_f1:
                best_val_f1    = f1
                best_threshold = thresh
                torch.save(
                    model.state_dict(),
                    os.path.join(OUTPUT_DIR, 'model_weights_best.pth'),
                )
                print(f'  New best checkpoint saved '
                      f'(thresh={best_threshold}, F1={best_val_f1:.3f})')

    # ── Save final model ──────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_weights.pth'))
    tokenizer.save_pretrained(OUTPUT_DIR)

    if not FINAL_SUBMISSION:
        with open(os.path.join(OUTPUT_DIR, 'best_threshold.txt'), 'w') as f:
            f.write(str(best_threshold))
        print(f'\nBest threshold: {best_threshold}  (val F1={best_val_f1:.3f})')
        print('Flip FINAL_SUBMISSION=True and retrain to use train+val combined.')
    else:
        print(f'\nFinal model saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
