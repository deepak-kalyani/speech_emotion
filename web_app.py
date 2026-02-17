import streamlit as st
import torch
import numpy as np

from features import extract_mfcc
from model import EmotionCNN
from rnn_model import EmotionRNN
from gru_model import EmotionGRU
from lstm_model import EmotionLSTM

# -----------------------------
# CONFIG
# -----------------------------
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

    cnn = EmotionCNN().to(device)
    cnn.load_state_dict(torch.load("results/best_emotion_cnn.pth", map_location=device))
    cnn.eval()
    models["CNN"] = cnn

    rnn = EmotionRNN().to(device)
    rnn.load_state_dict(torch.load("results_rnn/best_rnn.pth", map_location=device))
    rnn.eval()
    models["RNN"] = rnn

    lstm = EmotionLSTM().to(device)
    lstm.load_state_dict(torch.load("results_lstm/best_lstm.pth", map_location=device))
    lstm.eval()
    models["LSTM"] = lstm

    gru = EmotionGRU().to(device)
    gru.load_state_dict(torch.load("results_gru/best_gru.pth", map_location=device))
    gru.eval()
    models["GRU"] = gru

    return models

models = load_models()

# -----------------------------
# FEATURE PREP
# -----------------------------
def prepare_input(audio_path, model_type):
    mfcc = extract_mfcc(audio_path)

    # Ensure 2D MFCC
    if mfcc.ndim == 1:
        mfcc = np.expand_dims(mfcc, axis=1)

    mfcc = mfcc.T  # (T, 40)

    if model_type == "CNN":
        # CNN expects (batch, channel, 40)
        mfcc = mfcc.mean(axis=0)          # (40,)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        mfcc = mfcc.unsqueeze(0).unsqueeze(1)  # (1, 1, 40)

    else:
        # RNN / LSTM / GRU expect (batch, T, 40)
        MAX_LEN = 200
        if mfcc.shape[0] > MAX_LEN:
            mfcc = mfcc[:MAX_LEN]
        else:
            pad = MAX_LEN - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)))

        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        mfcc = mfcc.unsqueeze(0)  # (1, T, 40)

    return mfcc.to(device)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Speech Emotion Comparison", layout="wide")
st.title("🎧 Speech Emotion Recognition")
st.subheader("CNN vs RNN vs LSTM vs GRU")

mode = st.radio(
    "Choose Mode",
    ["Compare Models", "Select Model"]
)

audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

# -----------------------------
# MODE 1: COMPARE MODELS
# -----------------------------
if mode == "Compare Models" and audio_file:
    st.markdown("### 🔍 Model Comparison Results")

    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    cols = st.columns(4)

    for idx, (name, model) in enumerate(models.items()):
        with cols[idx]:
            inp = prepare_input("temp.wav", name)

            with torch.no_grad():
                output = model(inp)
                probs = torch.softmax(output, dim=1)[0]

            pred = torch.argmax(probs).item()
            confidence = probs[pred].item() * 100

            st.markdown(f"### {name}")
            st.write(f"**Emotion:** {EMOTIONS[pred]}")
            st.write(f"**Confidence:** {confidence:.2f}%")

# -----------------------------
# MODE 2: SELECT MODEL
# -----------------------------
elif mode == "Select Model":
    selected_model = st.selectbox("Choose Model", list(models.keys()))

    if audio_file:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        inp = prepare_input("temp.wav", selected_model)

        with torch.no_grad():
            output = models[selected_model](inp)
            probs = torch.softmax(output, dim=1)[0]

        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100

        st.success(f"🎯 **Emotion:** {EMOTIONS[pred]}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")
