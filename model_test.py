from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import random

# Fine-tune edilmiş model ve tokenizer
model_path = "D:/Data Mining/end-to-end-text-summarizer/fine_tuned_pegasus"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize edilmiş veri seti
tokenized_datasets = load_from_disk("D:/Data Mining/end-to-end-text-summarizer/tokenized_datasets")

# Test veri setinin alımı
test_dataset = tokenized_datasets["test"]

# Modeli bir örnek üzerinde test etme
def generate_summary(example):
    # Giriş metni
    input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

    # Modelin özet üretmesi
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=4,  # Beam Search genişliği
        length_penalty=0.6,  # Daha kısa özetleri tercih etme
        early_stopping=True  # İdeal beam bulunduğunda dur
    )

    # Üretilen özetin metne dönüştürülmesi
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Orijinal giriş ve referans özet
    reference_summary = tokenizer.decode(example["labels"], skip_special_tokens=True)

    return input_text, generated_summary, reference_summary

# Rastgele bir örnek seçimi
random_index = random.randint(0, len(test_dataset) - 1)
example = test_dataset[random_index]

# Özet üretimi
input_text, generated_summary, reference_summary = generate_summary(example)


print("\nOrijinal Giriş Metni:")
print(input_text)
print("\nModelin Ürettiği Özet:")
print(generated_summary)
print("\nReferans Özet:")
print(reference_summary)
