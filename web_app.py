import streamlit as st
import torch
import numpy as np

from features import extract_features
from model import EmotionCNN
from cnn_lstm_model import CNNLSTMModel
from rnn_model import EmotionRNN
from gru_model import EmotionGRU
from lstm_model import EmotionLSTM

EMOTIONS = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    models = {}

    try:
        cnn = EmotionCNN().to(device)
        cnn.load_state_dict(torch.load("results/best_emotion_cnn.pth", map_location=device))
        cnn.eval()
        models["CNN"] = cnn
    except:
        pass

    try:
        cnn_lstm = CNNLSTMModel().to(device)
        cnn_lstm.load_state_dict(torch.load("results_cnn_lstm/best_model.pth", map_location=device))
        cnn_lstm.eval()
        models["CNN+LSTM"] = cnn_lstm
    except:
        pass

    try:
        rnn = EmotionRNN().to(device)
        rnn.load_state_dict(torch.load("results_rnn/best_rnn.pth", map_location=device))
        rnn.eval()
        models["RNN"] = rnn
    except:
        pass

    try:
        lstm = EmotionLSTM().to(device)
        lstm.load_state_dict(torch.load("results_lstm/best_lstm.pth", map_location=device))
        lstm.eval()
        models["LSTM"] = lstm
    except:
        pass

    try:
        gru = EmotionGRU().to(device)
        gru.load_state_dict(torch.load("results_gru/best_gru.pth", map_location=device))
        gru.eval()
        models["GRU"] = gru
    except:
        pass

    return models

models = load_models()

# -----------------------------
# FEATURE PREP
# -----------------------------
def prepare_input(audio_path, model_name):
    features = extract_features(audio_path)  # (174, 128)

    if model_name in ["CNN", "CNN+LSTM"]:
        # These expect (1, 174, 128)
        tensor = torch.tensor(features, dtype=torch.float32)
        return tensor.unsqueeze(0).to(device)

    else:
        # RNN / LSTM / GRU expect (1, time, 40)
        mfcc = features[:40, :]              # (40, 128) — just MFCCs
        mfcc = mfcc.T                        # (128, 40)
        tensor = torch.tensor(mfcc, dtype=torch.float32)
        return tensor.unsqueeze(0).to(device)  # (1, 128, 40)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")
st.title("🎧 Speech Emotion Recognition")
st.subheader("Compare multiple deep learning models")

if not models:
    st.warning("⚠️ No trained models found. Please run train.py first.")
    st.stop()

st.write(f"✅ Loaded models: {', '.join(models.keys())}")

mode = st.radio("Choose Mode", ["Compare Models", "Select Model"])

audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

# -----------------------------
# MODE 1: COMPARE MODELS
# -----------------------------
if mode == "Compare Models" and audio_file:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    st.markdown("### 🔍 Model Comparison Results")
    cols = st.columns(len(models))

    for idx, (name, model) in enumerate(models.items()):
        with cols[idx]:
            try:
                inp = prepare_input("temp.wav", name)
                with torch.no_grad():
                    probs = torch.softmax(model(inp), dim=1)[0]
                pred       = torch.argmax(probs).item()
                confidence = probs[pred].item() * 100

                st.markdown(f"### {name}")
                st.write(f"**Emotion:** {EMOTIONS[pred]}")
                st.write(f"**Confidence:** {confidence:.2f}%")
            except Exception as e:
                st.markdown(f"### {name}")
                st.error(f"Error: {e}")

# -----------------------------
# MODE 2: SELECT MODEL
# -----------------------------
elif mode == "Select Model":
    selected = st.selectbox("Choose Model", list(models.keys()))

    if audio_file:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        try:
            inp = prepare_input("temp.wav", selected)
            with torch.no_grad():
                probs = torch.softmax(models[selected](inp), dim=1)[0]

            pred       = torch.argmax(probs).item()
            confidence = probs[pred].item() * 100

            st.success(f"🎯 **Emotion:** {EMOTIONS[pred]}")
            st.info(f"📊 **Confidence:** {confidence:.2f}%")

            # Show all emotion probabilities
            st.markdown("### 📊 All Emotion Probabilities")
            for i, emotion in enumerate(EMOTIONS):
                st.progress(float(probs[i].item()), text=f"{emotion}: {probs[i].item()*100:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")