import torch
from torchvision import models, transforms
from PIL import Image
import os

# 🔧 Load model (must match what you trained)
image_model = models.shufflenet_v2_x0_5(pretrained=False)
image_model.fc = torch.nn.Linear(1024, 7)  # 7 emotion classes
image_model.load_state_dict(torch.load("models/image_emotion_model/shufflenet_fast.pth", map_location="cpu"))
image_model.eval()

# ✅ Define class names (same as your dataset folders)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 📷 Load and preprocess test image
image_path = "test_sample.png"  # 🔁 Replace with your test image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # simple normalization
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # add batch dimension

# 🔮 Predict
with torch.no_grad():
    outputs = image_model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

print(f"✅ Predicted emotion: {predicted_class}")
