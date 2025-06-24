import torch
from torchvision import models, transforms
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Image preprocessing
image_path = "test_sample.png"  # make sure this file exists in the same folder
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image_tensor = transform(image).unsqueeze(0).to(device)

# ✅ Load Image Model (ShuffleNet)
num_classes = 7
image_model = models.shufflenet_v2_x0_5(pretrained=True)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, num_classes)
state_dict = torch.load("models/image_emotion_model/shufflenet_fast.pth", map_location=device)
image_model.load_state_dict(state_dict)
image_model.to(device)
image_model.eval()

# ✅ Load Text Model
tokenizer = DistilBertTokenizerFast.from_pretrained("models/text_emotion_model")
text_model = DistilBertForSequenceClassification.from_pretrained("models/text_emotion_model")
text_model.to(device)
text_model.eval()

# ✅ Sample text input
text = "I feel so happy and grateful for everything today"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

# ✅ Predict
with torch.no_grad():
    # Image prediction
    image_output = image_model(image_tensor)
    image_probs = F.softmax(image_output, dim=1)

    # Text prediction
    text_output = text_model(**inputs)
    text_probs = F.softmax(text_output.logits, dim=1)

# ✅ Combine predictions
combined_probs = (image_probs + text_probs) / 2
predicted_class = torch.argmax(combined_probs, dim=1).item()

# ✅ Emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'love', 'neutral', 'sadness', 'surprise']
print(f"🧠 Text Prediction: {emotion_labels[torch.argmax(text_probs)]}")
print(f"🖼️ Image Prediction: {emotion_labels[torch.argmax(image_probs)]}")
print(f"🔮 Final Multimodal Prediction: {emotion_labels[predicted_class]}")
