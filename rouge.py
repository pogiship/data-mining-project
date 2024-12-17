import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate

model_path = "D:/Data Mining/end-to-end-text-summarizer/fine_tuned_pegasus"
data_path = "D:/Data Mining/end-to-end-text-summarizer/tokenized_datasets"

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# Model ve tokenizer'ı yükleme
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test veri setini yükleme
tokenized_datasets = load_from_disk(data_path)
test_dataset = tokenized_datasets["test"]

# ROUGE metriğini yükleme
rouge = evaluate.load("rouge")

# Test veri setinden özetleme ve ROUGE hesaplama
def evaluate_model(model, tokenizer, test_dataset, rouge, num_samples=100):
    predictions = []
    references = []

    for example in test_dataset.select(range(num_samples)):  # Belirli sayıda örnek üzerinde değerlendirme
        # Giriş metnini al
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

        # Modelden özet üret
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=128,  # Özetin maksimum uzunluğu
            min_length=20,   # Özetin minimum uzunluğu
            num_beams=4,     # Beam Search genişliği
            length_penalty=0.6,
            early_stopping=True
        )
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Referans özeti al
        reference_summary = tokenizer.decode(example["labels"], skip_special_tokens=True)

        # Listelere ekle
        predictions.append(generated_summary)
        references.append(reference_summary)

    # ROUGE metriğini hesapla
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return results

# Modelin evaluationu
rouge_scores = evaluate_model(model, tokenizer, test_dataset, rouge, num_samples=100)  # İlk 100 örnek için değerlendirme

# Sonuç
print("ROUGE Skorları:")
for key, value in rouge_scores.items():
    print(f"{key}: {value:.4f}")
