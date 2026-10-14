from config_asc import ID2LABEL

def compute_metrics(y_true, y_pred):
    labels = sorted(ID2LABEL.keys())
    n = len(labels)
    confusion = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(y_true, y_pred):
        confusion[int(gold)][int(pred)] += 1
    total = len(y_true)
    correct = sum(confusion[i][i] for i in labels)
    accuracy = correct / total if total else 0.0
    per_class = {}
    precisions, recalls, f1s = [], [], []
    weighted_f1_sum = 0.0
    for class_id in labels:
        name = ID2LABEL[class_id]
        tp = confusion[class_id][class_id]
        fp = sum(confusion[r][class_id] for r in labels if r != class_id)
        fn = sum(confusion[class_id][c] for c in labels if c != class_id)
        support = sum(confusion[class_id])
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {'precision': p, 'recall': r, 'f1': f1, 'support': support, 'tp': tp, 'fp': fp, 'fn': fn}
        precisions.append(p); recalls.append(r); f1s.append(f1)
        weighted_f1_sum += f1 * support
    return {
        'accuracy': accuracy,
        'macro_precision': sum(precisions) / n,
        'macro_recall': sum(recalls) / n,
        'macro_f1': sum(f1s) / n,
        'weighted_f1': weighted_f1_sum / total if total else 0.0,
        'per_class': per_class,
        'confusion_matrix': confusion,
        'confusion_matrix_labels': [ID2LABEL[i] for i in labels],
    }
