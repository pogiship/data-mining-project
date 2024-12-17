from transformers import PegasusTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import json

# Importing the file paths
train_file = "D:/Data Mining/end-to-end-text-summarizer/dataset/corpus/train.json"
val_file = "D:/Data Mining/end-to-end-text-summarizer/dataset/corpus/val.json"
test_file = "D:/Data Mining/end-to-end-text-summarizer/dataset/corpus/test.json"

# Loading JSON files
with open(train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(val_file, "r", encoding="utf-8") as f:
    val_data = json.load(f)

with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Convert datasets into Hugging Face `Dataset` format
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

# Combine datasets into a `DatasetDict` format
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Load the tokenizer
model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize target text (summary)
def preprocess_function(batch):
    model_inputs = tokenizer(
        batch["dialogue"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["summary"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization to all datasets
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# Define a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

# Check an example from the tokenized training data
print("Tokenize edilmiş eğitim verisinden bir örnek:", tokenized_datasets["train"][0])
# Save the tokenized datasets to disk
tokenized_datasets.save_to_disk("D:/Data Mining/end-to-end-text-summarizer/tokenized_datasets")
