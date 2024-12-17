import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk

def train_model():

    model_path = "D:/Data Mining/end-to-end-text-summarizer\pegasus_cnn_model"

    # GPU kontrolü
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")

    # Model ve tokenizer'ı yükleme
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)  # Model GPU'ya taşınıyor
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize edilmiş veri setini yükleme
    data_path = "D:/Data Mining/end-to-end-text-summarizer/tokenized_datasets"
    tokenized_datasets = load_from_disk(data_path)

    # Data collator tanımlama
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # TrainingArguments hyperparametreleri
    training_args = TrainingArguments(
        output_dir="D:/Data Mining/end-to-end-text-summarizer/results",  # Çıktıların kaydedileceği dizin
        num_train_epochs=5,                                              # Eğitim sırasında toplam 5 epoch
        warmup_steps=500,                                                # Öğrenme oranı ısınması için 500 adım
        per_device_train_batch_size=3,                                   # Her bir cihaz için eğitim batch boyutu
        per_device_eval_batch_size=3,                                    # Her bir cihaz için değerlendirme batch boyutu
        weight_decay=0.01,                                               # Ağırlık azalması oranı
        logging_steps=100,                                               # Her 100 adımda bir loglama
        evaluation_strategy="steps",                                     # Belirli adımlarda değerlendirme yapılacak
        eval_steps=500,                                                  # Her 500 adımda bir değerlendirme
        save_steps=1000,                                                 # Model 1000 adımda bir kaydedilecek
        gradient_accumulation_steps=16,                                  # 16 adım boyunca gradyan birikimi yapılacak
        save_total_limit=2,                                              # En fazla 2 checkpoint sakla
        fp16=True                                                        # Daha hızlı işlem için 16-bit floating point kullan
    )


    trainer = Trainer(
        model=model,                           # Model (GPU'da)
        args=training_args,                    # Eğitim argümanları
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # train başlatma
    trainer.train()

    # Modeli ve tokenizer'ı kaydetme
    model.save_pretrained("D:/Data Mining/end-to-end-text-summarizer/fine_tuned_pegasus")
    tokenizer.save_pretrained("D:/Data Mining/end-to-end-text-summarizer/fine_tuned_pegasus")


# Multiprocessing hatalarını önlemek için ana modül kontrolü
if __name__ == "__main__":
    train_model()
