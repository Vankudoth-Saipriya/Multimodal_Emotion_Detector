# train_image_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader, Subset

# ✅ Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ✅ Paths
data_dir = "image_data"  # assumes split_val.py already created train/val folders
model_path = "models/image_emotion_model"
os.makedirs(model_path, exist_ok=True)

# ✅ Classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ✅ Transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

# ✅ Load datasets (using subset of training data to reduce time)
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

# ✅ Reduce dataset size for faster training
subset_size = 35000 # max training images
if len(train_dataset) > subset_size:
    train_dataset = Subset(train_dataset, list(range(subset_size)))

# ✅ DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"📁 Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# ✅ Model
model = models.shufflenet_v2_x0_5(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\n🔁 Epoch {epoch+1}/{epochs}")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 5 == 0:
            print(f"   🔄 Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    acc = 100. * correct / total
    print(f"✅ Epoch {epoch+1} complete | Train Accuracy: {acc:.2f}% | Loss: {total_loss / len(train_loader):.4f}")

    # ✅ Evaluate on validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100. * correct_val / total_val
    print(f"📊 Validation Accuracy after Epoch {epoch+1}: {val_acc:.2f}%")

# ✅ Save Model
torch.save(model.state_dict(), os.path.join(model_path, "shufflenet_fast.pth"))
print(f"\n✅ Model saved to {model_path}/shufflenet_fast.pth")
