


import torchaudio
import streamlit
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer
import soundfile as sf
import numpy as np

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load emotion labels
image_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
text_labels = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

# Load models
@st.cache_resource
def load_image_model():
    model = models.shufflenet_v2_x0_5(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(image_labels))
    model.load_state_dict(torch.load("models/image_emotion_model/shufflenet_fast.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_text_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/text_emotion_model")
    model = DistilBertForSequenceClassification.from_pretrained("models/text_emotion_model")
    model.to(device)
    model.eval()
    return tokenizer, model

image_model = load_image_model()
tokenizer, text_model = load_text_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("🎭 Emotion Detection App")

option = st.radio("Choose Input Type:", ["Text", "Image", "Audio"])

if option == "Text":
    text_input = st.text_area("Enter a sentence to analyze emotion:")
    if st.button("Predict Text Emotion"):
        if text_input.strip():
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = text_model(**inputs)
                probs = F.softmax(output.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            st.success(f"🧠 **Predicted Emotion:** {text_labels[pred]}")
        else:
            st.warning("Please enter some text.")

elif option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_tensor = transform(image).unsqueeze(0).to(device)

        if st.button("Predict Image Emotion"):
            with torch.no_grad():
                output = image_model(image_tensor)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            st.success(f"🖼️ **Predicted Emotion:** {image_labels[pred]}")

elif option == "Audio":
    st.header("🎤 Audio Emotion Detection")

    audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
    
    if audio_file is not None:
        # Save and load audio
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        # Load model
        model_name = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        # Load audio
        audio_input, sample_rate = librosa.load("temp.wav", sr=16000)
        inputs = tokenizer(audio_input, return_tensors="pt", sampling_rate=16000, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        labels = ["angry", "happy", "neutral", "sad"]  # Labels for this model
        st.success(f"🎧 Predicted Emotion: **{labels[predicted_class]}**")
