from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "models/text_emotion_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Sample input
text = "I am feeling so sad and alone today."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

# Reverse label map
id2label = model.config.id2label
print(f"Predicted emotion: {id2label[predicted_class_id]}")

sentences = [
    "I am so happy today!",
    "Why is everything going wrong?",
    "I'm scared about the exam.",
    "I miss my family."
]

for text in sentences:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    print(f"Text: {text} --> Emotion: {predicted_label}")
