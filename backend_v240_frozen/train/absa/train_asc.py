import json
import math
import random
from collections import Counter
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from config_asc import (
    TRAIN_FILE, VAL_FILE, MODEL_OUTPUT_DIR, MODEL_NAME, LABEL2ID, ID2LABEL,
    BATCH_SIZE, GRAD_ACCUM_STEPS, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_RATIO, EARLY_STOPPING_PATIENCE, RANDOM_SEED, DEVICE,
)
from dataset_asc import ABSASentimentDataset
from metrics_asc import compute_metrics

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def calculate_class_weights():
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = Counter(row['label'] for row in data)
    max_count = max(counts.values())
    weights = []
    for class_id in range(len(ID2LABEL)):
        label = ID2LABEL[class_id]
        count = max(counts.get(label, 0), 1)
        weights.append(math.sqrt(max_count / count))
    print('ASC label counts:', dict(counts))
    print('ASC class weights:', {ID2LABEL[i]: weights[i] for i in range(len(weights))})
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(loader) if len(loader) else 0.0
    return metrics

def main():
    set_seed(RANDOM_SEED)
    print('=' * 70)
    print('PhoBERT ASC V1')
    print('=' * 70)
    print('Device:', DEVICE)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ABSASentimentDataset(TRAIN_FILE, tokenizer)
    val_dataset = ABSASentimentDataset(VAL_FILE, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=calculate_class_weights())
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_training_steps = update_steps_per_epoch * EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )
    history = []
    best_macro_f1 = -1.0
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch {epoch}/{EPOCHS}')
        model.train(); optimizer.zero_grad(); total_train_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            (loss / GRAD_ACCUM_STEPS).backward()
            total_train_loss += loss.item()
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 10 == 0:
                print(f'  step {step}/{len(train_loader)} loss={loss.item():.4f}')
        avg_train_loss = total_train_loss / len(train_loader)
        metrics = evaluate(model, val_loader, criterion)
        epoch_result = {
            'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': metrics['loss'],
            'accuracy': metrics['accuracy'], 'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'], 'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'], 'per_class': metrics['per_class'],
            'confusion_matrix': metrics['confusion_matrix'],
            'confusion_matrix_labels': metrics['confusion_matrix_labels'],
        }
        history.append(epoch_result)
        save_json(MODEL_OUTPUT_DIR / 'training_history.json', history)
        print(f"Train loss : {avg_train_loss:.4f}")
        print(f"Val loss   : {metrics['loss']:.4f}")
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"Macro F1   : {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        for label_name, row in metrics['per_class'].items():
            print(f"  {label_name:8} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} N={row['support']}")
        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']; patience = 0
            print('[+] New best ASC model')
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
            save_json(MODEL_OUTPUT_DIR / 'best_validation_metrics.json', epoch_result)
        else:
            patience += 1
            print(f'No improvement ({patience}/{EARLY_STOPPING_PATIENCE})')
        if patience >= EARLY_STOPPING_PATIENCE:
            print('[!] Early stopping')
            break
    print('\n' + '=' * 70)
    print(f'Best validation Macro-F1: {best_macro_f1:.4f}')
    print('Saved:', MODEL_OUTPUT_DIR)

if __name__ == '__main__':
    main()
