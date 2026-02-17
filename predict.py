import torch
from features import extract_mfcc
from model import EmotionCNN

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("results/best_emotion_cnn.pth"))
model.eval()

# -----------------------------
# EMOTION LABELS
# -----------------------------
emotion_map = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# -----------------------------
# AUDIO FILE TO PREDICT
# -----------------------------
audio_path = "sample.wav"   

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
mfcc = extract_mfcc(audio_path)
mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# shape: (1, 1, 40)

# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():
    output = model(mfcc)
    pred_class = torch.argmax(output, dim=1).item()

print("🎵 Audio:", audio_path)
print("🧠 Predicted Emotion:", emotion_map[pred_class])
