import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset



# ✅ Load dataset
def load_data(path):
    df = pd.read_csv(path, sep=';', names=["text", "label"])
    
    print(f"Loaded {len(df)} rows from {path}")
    print("First few raw labels:", df["label"].unique()[:10])
    
    label_map = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'love': 4,
        'neutral': 5,
        'sadness': 6,
        'surprise': 7
    }
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(label_map)



    print(f"Loaded {len(df)} rows from {path}")
    print("NaNs in label column:", df["label"].isna().sum())


    df = df.dropna(subset=["text", "label"]) 
    return df

train_df = load_data("train.txt")
val_df = load_data("val.txt")

print("Train df shape:", train_df.shape)
print("Train df head:\n", train_df.head())

print(train_df.head())
print("Labels:", train_df['label'].unique())


# ✅ Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


train_df["label"] =  train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int) 

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)




# ✅ Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))


train_dataset = train_dataset.map(tokenize_function, batched=True)


val_dataset = val_dataset.map(tokenize_function, batched=True)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))




train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print(train_dataset[0].keys())  # This will work now


# ✅ Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./text_model_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch"
)

# ✅ Trainer
from transformers import Trainer
import torch.nn as nn

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ✅ Train and Save
trainer.train()
model.save_pretrained("../models/text_emotion_model")
tokenizer.save_pretrained("../models/text_emotion_model")


train_df.to_csv("debug_cleaned_train.csv", index=False)

