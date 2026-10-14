from transformers import AutoModel, AutoTokenizer

print("[*] Downloading PhoBERT...")

model = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

print("[*] Saving model...")

model.save_pretrained("./model/phobert")
tokenizer.save_pretrained("./model/phobert")

print("[+] DONE")