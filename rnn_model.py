import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = "data/RAVDESS"
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001
NUM_CLASSES = 8
PATIENCE = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results_rnn"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_ACTORS = [f"Actor_{i:02d}" for i in range(1, 19)]
TEST_ACTORS  = [f"Actor_{i:02d}" for i in range(19, 25)]

def pad_or_truncate(mfcc, max_len=MAX_LEN):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    else:
        pad_width = max_len - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")

# =====================================================
# MODEL ← USED BY STREAMLIT
# =====================================================
class EmotionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_MFCC,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(128 * 2, NUM_CLASSES)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)

# =====================================================
# TRAINING (ONLY WHEN RUN DIRECTLY)
# =====================================================
if __name__ == "__main__":
    from features import extract_features

    class RNNEmotionDataset(Dataset):
        def __init__(self, actors):
            self.samples = []
            for actor in actors:
                actor_path = os.path.join(DATASET_PATH, actor)
                for file in os.listdir(actor_path):
                    if file.endswith(".wav"):
                        path = os.path.join(actor_path, file)
                        label = int(file[6:8]) - 1
                        self.samples.append((path, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            mfcc = extract_features(path)   # (174, 128)
            mfcc = mfcc.T                   # (128, 174)
            mfcc = pad_or_truncate(mfcc, MAX_LEN)
            return (
                torch.tensor(mfcc, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            )

    print("Using device:", device)

    train_loader = DataLoader(RNNEmotionDataset(TRAIN_ACTORS), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(RNNEmotionDataset(TEST_ACTORS),  batch_size=BATCH_SIZE)

    model     = EmotionRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, wait = 0, 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        y_true, y_pred, val_loss = [], [], 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())

        val_loss /= len(test_loader)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_rnn.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⏹ Early stopping")
                break

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_rnn.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X.to(device))
            y_true.extend(y.numpy())
            y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())

    print(classification_report(y_true, y_pred))
    with open(f"{RESULTS_DIR}/metrics_rnn.txt", "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}\n\n")
        f.write(classification_report(y_true, y_pred))
    print("✅ RNN results saved in 'results_rnn/'")