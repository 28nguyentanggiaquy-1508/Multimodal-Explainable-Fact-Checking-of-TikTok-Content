import json
import os


class ContextDictionary:
    def __init__(self, dict_path="assets/dic_final_cleaned.json"):
        self.dictionary = {}

        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)

            print(f"[+] Loaded dictionary: {dict_path}")
        else:
            print("[!] Dictionary file not found")

    def detect_emotion_boost(self, text: str):
        text = text.lower()

        scores = {
            "ANGER": 0,
            "DISGUST": 0,
            "ENJOYMENT": 0,
            "FEAR": 0,
            "OTHER": 0,
            "SADNESS": 0,
            "SURPRISE": 0,
        }

        mapping = {
            "positive": "ENJOYMENT",
            "negative": "ANGER",
            "sadness": "SADNESS",
            "surprise": "SURPRISE"
        }

        for group_name, words in self.dictionary.items():
            label = mapping.get(group_name)

            if not label:
                continue

            for word in words:
                if word.lower() in text:
                    scores[label] += 1

        return scores