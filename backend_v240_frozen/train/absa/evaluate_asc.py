import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config_asc import TEST_FILE, MODEL_OUTPUT_DIR, BATCH_SIZE, DEVICE
from dataset_asc import ABSASentimentDataset
from metrics_asc import compute_metrics

@torch.no_grad()
def main():
    print('=' * 70)
    print('PhoBERT ASC V1 - TEST EVALUATION')
    print('=' * 70)
    print('Device:', DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR).to(DEVICE)
    model.eval()
    dataset = ABSASentimentDataset(TEST_FILE, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_true, y_pred = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    print(f"Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Macro P     : {metrics['macro_precision']:.4f}")
    print(f"Macro R     : {metrics['macro_recall']:.4f}")
    print(f"Macro F1    : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 : {metrics['weighted_f1']:.4f}")
    print('\nPer-class:')
    for label, row in metrics['per_class'].items():
        print(f"{label:8} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} N={row['support']}")
    print('\nConfusion matrix labels:', metrics['confusion_matrix_labels'])
    for row in metrics['confusion_matrix']:
        print(row)
    output_file = MODEL_OUTPUT_DIR / 'test_metrics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('\nSaved:', output_file)

if __name__ == '__main__':
    main()
