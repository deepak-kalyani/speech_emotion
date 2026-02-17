import torch
from features import extract_features
from model import EmotionCNN

EMOTION_MAP = {
    0: "Neutral", 1: "Calm",    2: "Happy",    3: "Sad",
    4: "Angry",   5: "Fearful", 6: "Disgust",  7: "Surprised"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("results/best_emotion_cnn.pth", map_location=device))
model.eval()

def predict_emotion(audio_path):
    features = extract_features(audio_path)                          # (174, 128)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    # shape: (1, 174, 128) — batch of 1

    with torch.no_grad():
        output = model(features)
        probs  = torch.softmax(output, dim=1)
        pred   = torch.argmax(probs, dim=1).item()

    return EMOTION_MAP[pred], probs[0][pred].item()