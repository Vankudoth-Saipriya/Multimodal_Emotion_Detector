# Multimodal Emotion Detection System

A lightweight, easy-to-use emotion detection system that predicts emotions from either **text**, **image**, or **audio** input — all handled via a simple **Streamlit web app**.

🌐 **Live Demo**: [https://multimodalemotiondetection.streamlit.app/](https://multimodalemotiondetection.streamlit.app/)

---

## 🚀 Features

- 📄 **Text Emotion Detection** – Uses DistilBERT to detect emotions from sentences.
- 🖼️ **Image Emotion Detection** – Uses a custom-trained ShuffleNet model on facial expression images.
- 🎤 **Audio Emotion Detection** – Uses Wav2Vec2 model to classify emotion from audio clips.
- 📦 **Streamlit UI** – Upload interface for text, image, and audio.

> ⚠️ **Note:** Live camera and audio recording features are not yet implemented — only file upload is currently supported.

---

## 🔍 Use Cases

- Customer feedback sentiment analysis
- Emotion detection for healthcare/chatbots
- Social media content analysis

---

## 🧠 Models Used

| Modality | Model         | Source                                     |
| -------- | ------------- | ------------------------------------------ |
| Text     | DistilBERT    | HuggingFace Transformers                   |
| Image    | ShuffleNet V2 | Custom trained on FER+ dataset             |
| Audio    | Wav2Vec2      | Fine-tuned from pre-trained Facebook model |

---

## 📁 Project Structure

```
multimodal_emotion_detection/
├── app.py                         # Streamlit app
├── models/
│   ├── text_emotion_model/       # DistilBERT tokenizer + weights
│   └── image_emotion_model/      # ShuffleNet image model (shufflenet_fast.pth)
├── text_model/
│   └── train_text_model.py       # Text model training script
├── predict_text.py               # Text prediction logic
├── predict_image_only.py         # Image prediction logic
├── predict_multimodal.py         # Combined inference (not used in current version)
├── requirements.txt              # Dependencies
└── README.md                     # Project info
```

---

## 🛠️ Setup Instructions

1. **Clone the repo**:

```bash
git clone https://github.com/Vankudoth-Saipriya/multimodal_emotion_detection.git
cd multimodal_emotion_detection
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run Streamlit App**:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- ✅ Add webcam-based image detection
- ✅ Add real-time microphone recording for audio
- ⏳ Add sentiment fusion model (text + image + audio)
- 🌐 Mobile responsiveness

---

## 📦 Deployment

- App is deployed on **Streamlit Cloud**.
- Requirements are auto-installed from `requirements.txt`.
- Model weights are loaded from GitHub (Git LFS enabled).

---

## 🙌 Acknowledgments

- HuggingFace 🤗 for pretrained models
- PyTorch and TorchVision
- Streamlit for UI simplicity
- FER+ dataset for image emotion training

---

## 👤 Author

**Sai Priya Vankudoth**

- [GitHub](https://github.com/Vankudoth-Saipriya)

---

## 📌 License

This project is under the MIT License.

