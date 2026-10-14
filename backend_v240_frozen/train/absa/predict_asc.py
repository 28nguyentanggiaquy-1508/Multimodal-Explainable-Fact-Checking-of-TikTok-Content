import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config_asc import MODEL_OUTPUT_DIR, MAX_LENGTH, DEVICE, ID2LABEL

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model

@torch.no_grad()
def predict_sentiment(text, aspect, tokenizer, model):
    encoded = tokenizer(
        text, aspect, truncation='only_first', max_length=MAX_LENGTH,
        padding=True, return_tensors='pt'
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    pred_id = int(logits.argmax(dim=-1)[0].item())
    return {
        'text': text,
        'aspect': aspect,
        'label': ID2LABEL[pred_id],
        'confidence': probs[pred_id],
        'probabilities': {ID2LABEL[i]: probs[i] for i in range(len(probs))},
    }

def main():
    tokenizer, model = load_model()
    print('PhoBERT ASC V1')
    print('Device:', DEVICE)
    while True:
        text = input('\nComment: ').strip()
        if not text:
            break
        aspect = input('Aspect : ').strip()
        if not aspect:
            print('Aspect không được để trống.')
            continue
        result = predict_sentiment(text, aspect, tokenizer, model)
        print('Label:', result['label'])
        print('Confidence:', round(result['confidence'], 4))
        print('Probabilities:', {k: round(v, 4) for k, v in result['probabilities'].items()})

if __name__ == '__main__':
    main()
