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

from features import extract_mfcc

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

# =====================================================
# SPEAKER-INDEPENDENT SPLIT
# =====================================================
TRAIN_ACTORS = [f"Actor_{i:02d}" for i in range(1, 19)]
TEST_ACTORS  = [f"Actor_{i:02d}" for i in range(19, 25)]

# =====================================================
# UTILS
# =====================================================
def pad_or_truncate(mfcc, max_len=MAX_LEN):
    if mfcc.shape[0] > max_len:
        return mfcc[:max_len, :]
    else:
        pad = max_len - mfcc.shape[0]
        return np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")

# =====================================================
# DATASET
# =====================================================
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
        mfcc = extract_mfcc(path)

        if mfcc.ndim == 1:
            mfcc = np.expand_dims(mfcc, axis=1)

        mfcc = mfcc.T
        mfcc = pad_or_truncate(mfcc)

        return (
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# =====================================================
# MODEL (Bi-LSTM)  ← USED BY STREAMLIT
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
# TRAINING (ONLY WHEN FILE IS RUN DIRECTLY)
# =====================================================
if __name__ == "__main__":

    print("Using device:", device)

    train_loader = DataLoader(
        LSTMDataset(TRAIN_ACTORS),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        LSTMDataset(TEST_ACTORS),
        batch_size=BATCH_SIZE
    )

    model = EmotionLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_f1 = 0
    wait = 0

    # ================= TRAIN LOOP =================
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
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / len(train_loader)
        train_acc = correct / total

        model.eval()
        correct, total, loss_sum = 0, 0, 0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                probs = torch.softmax(out, dim=1)

                loss_sum += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.argmax(1).cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        val_loss = loss_sum / len(test_loader)
        val_acc = correct / total
        f1 = f1_score(y_true, y_pred, average="weighted")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc {train_acc:.2f} | Val Acc {val_acc:.2f} | F1 {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            wait = 0
            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_lstm.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("⏹ Early stopping")
                break

    # ================= PLOTS =================
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
    plt.close()

    # ================= METRICS =================
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred))

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()

    # ================= ROC-AUC =================
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), np.array(y_prob).ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/roc_auc.png")
    plt.close()

    print("✅ LSTM complete. All results saved in results_lstm/")
