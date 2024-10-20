import json
from transformers import pipeline

with open('../config/config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

with open('../data/banking_dialogues.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
classifier = pipeline("zero-shot-classification", model=config["model_name"])

# Intentler (sınıflar) listesi
candidate_labels = config["candidate_labels"]

for dialogue in data:
    text = " ".join([f"{turn['customer']}" for turn in dialogue['conversation'] if 'customer' in turn])
    text += " " + " ".join([f"{turn['agent']}" for turn in dialogue['conversation'] if 'agent' in turn])

    # Zero-shot intent tahmini
    result = classifier(text, candidate_labels)

    print("Diyalog ID:", dialogue["id"])
    print("Metin:", text)
    print("Tahmin edilen intent:", result["labels"][0])
    print("Olasılıklarıyla birlikte tüm sonuçlar:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label}: {round(score, 4)}")
    print("\n" + "-" * 50 + "\n")
