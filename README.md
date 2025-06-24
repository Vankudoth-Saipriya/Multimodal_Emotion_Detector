🎭 Multimodal Emotion Detection

An intelligent emotion detection system capable of analyzing text, image, and audio inputs to detect human emotions with high accuracy using deep learning models.

🌐 Live Demo: https://multimodalemotiondetection.streamlit.app/

📌 Features

📄 Text Emotion Detection: Uses a fine-tuned DistilBERT model to classify emotions from text.

🖼️ Image Emotion Detection: Utilizes a CNN-based ShuffleNet model trained on facial expressions.

🔊 Audio Emotion Detection: Employs Wav2Vec2.0 model for speech-based emotion recognition.

📁 Simple UI: Upload audio (WAV), text, or image files to see real-time predictions.

🧠 Multimodal: Works on three modalities independently, allowing flexibility in user input.

📂 Project Structure

multimodal_emotion_detection/
├── app.py                            # Main Streamlit app
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   ├── image_emotion_model/
│   │   └── shufflenet_fast.pth      # Trained CNN model
│   └── text_emotion_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── ...
├── predict_image_only.py            # Standalone image prediction script
├── predict_text.py                  # Standalone text prediction script
├── predict_multimodal.py           # Old multimodal (optional)
├── text_model/
│   └── train_text_model.py          # Script to train text model
└── train_image_model.py             # Script to train image model

🚀 Quick Start

🖥️ Run Locally

Clone the Repository

git clone https://github.com/Vankudoth-Saipriya/multimodal_emotion_detection.git
cd multimodal_emotion_detection

Install Requirements

pip install -r requirements.txt

Launch App

streamlit run app.py

🛠️ Models Used

Modality

Model

Description

Text

DistilBERT

Fine-tuned for 6 emotion categories

Image

ShuffleNet Fast

Lightweight CNN trained on facial datasets

Audio

Wav2Vec2.0 (HuggingFace)

Emotion classification from raw waveform

📅 Usage (on Web App)

Text Tab

Paste or type text → Click "Predict Emotion"

Image Tab

Upload .jpg or .png file → See predicted emotion

Audio Tab

Upload .wav file → Audio gets classified

📝 Future Improvements

🔴 Add webcam capture for real-time facial emotion detection.

🔴 Add real-time audio recording for voice emotion input.

🟡 Expand datasets and refine model accuracies.

🟢 Integrate combined (fused) multimodal prediction pipeline.

🔵 Enable deployment on HuggingFace Spaces / Docker.

📦 Requirements

All dependencies are listed in requirements.txt. Major ones:

streamlit
torch
transformers
torchaudio
Pillow
opencv-python
librosa

👩‍💼 Author

Sai Priya Vankudoth

🌐 GitHub: @Vankudoth-Saipriya

🧪 License

This project is licensed under the MIT License — see the LICENSE file for details.

