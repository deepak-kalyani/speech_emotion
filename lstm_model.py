import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = "data/RAVDESS"
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
NUM_CLASSES = 8
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results_lstm"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_ACTORS = [f"Actor_{i:02d}" for i in range(1, 19)]
TEST_ACTORS  = [f"Actor_{i:02d}" for i in range(19, 25)]

def pad_or_truncate(mfcc, max_len=MAX_LEN):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    else:
        pad = max_len - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")

# =====================================================
# MODEL ← USED BY STREAMLIT
# =====================================================
class EmotionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_MFCC,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)

# =====================================================
# TRAINING (ONLY WHEN RUN DIRECTLY)
# =====================================================
if __name__ == "__main__":
    from features import extract_features

    class LSTMDataset(Dataset):
        def __init__(self, actors):
            self.samples = []
            for actor in actors:
                path = os.path.join(DATASET_PATH, actor)
                for file in os.listdir(path):
                    if file.endswith(".wav"):
                        label = int(file[6:8]) - 1
                        self.samples.append((os.path.join(path, file), label))

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

    train_loader = DataLoader(LSTMDataset(TRAIN_ACTORS), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(LSTMDataset(TEST_ACTORS),  batch_size=BATCH_SIZE)

    model     = EmotionLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, wait = 0, 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)

        train_loss = loss_sum / len(train_loader)

        model.eval()
        correct, total, loss_sum = 0, 0, 0
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out   = model(X)
                probs = torch.softmax(out, dim=1)
                loss_sum += criterion(out, y).item()
                correct  += (out.argmax(1) == y).sum().item()
                total    += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.argmax(1).cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        val_loss = loss_sum / len(test_loader)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | F1 {f1:.3f}")

        if f1 > best_f1:
            best_f1, wait = f1, 0
            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_lstm.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⏹ Early stopping")
                break

    model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_lstm.pth"))
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            out   = model(X.to(device))
            probs = torch.softmax(out, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(classification_report(y_true, y_pred))

    with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred))

    print("✅ LSTM complete. Results saved in results_lstm/")