import json
import torch
from torch.utils.data import Dataset
from config_asc import LABEL2ID, MAX_LENGTH

class ABSASentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoded = self.tokenizer(
            sample['text'],
            sample['aspect'],
            truncation='only_first',
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
        )
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(LABEL2ID[sample['label']], dtype=torch.long),
        }
