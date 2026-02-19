"""
Speech Emotion Recognition System
Copyright (c) 2026 Deepak Kalyani
Licensed under MIT License - see LICENSE file for details
GitHub: https://github.com/deepak-kalyani/speech_emotion
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from dataset import RAVDESSDataset
from model import EmotionCNN
from cnn_lstm_model import CNNLSTMModel

# -----------------------------
# CHOOSE YOUR MODEL HERE
# Options: 'cnn' or 'cnn_lstm'
# -----------------------------
MODEL_TYPE = 'cnn_lstm'

if MODEL_TYPE == 'cnn_lstm':
    RESULTS_DIR = "results_cnn_lstm"
else:
    RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Using model:", MODEL_TYPE)

# -----------------------------
# DATASET
# -----------------------------
use_augment = False  # augment only for new model
dataset = RAVDESSDataset("data/RAVDESS", augment=use_augment)
total_size = len(dataset)
print(f"Total samples (with augmentation): {total_size}")

train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# -----------------------------
# MODEL
# -----------------------------
if MODEL_TYPE == 'cnn_lstm':
    model = CNNLSTMModel().to(device)
else:
    model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# -----------------------------
# TRAINING LOOP
# -----------------------------
max_epochs = 100
patience   = 10
best_val_acc      = 0
epochs_no_improve = 0
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(max_epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (torch.max(outputs, 1)[1] == y).sum().item()
        total   += y.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            val_loss += criterion(outputs, y).item()
            correct  += (torch.max(outputs, 1)[1] == y).sum().item()
            total    += y.size(0)

    val_loss /= len(val_loader)
    val_acc   = 100 * correct / total

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{max_epochs} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc      = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model.pth")
        print(f"  ✅ New best model saved ({val_acc:.2f}%)")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("⏹ Early stopping triggered")
        break

# -----------------------------
# PLOTS
# -----------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.legend(); plt.title("Loss Curve")
plt.savefig(f"{RESULTS_DIR}/loss_curve.png"); plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs,   label="Val Acc")
plt.legend(); plt.title("Accuracy Curve")
plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png"); plt.close()

# -----------------------------
# TEST EVALUATION
# -----------------------------
model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_model.pth"))
model.eval()

y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        probs = torch.softmax(model(X), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(torch.argmax(probs, 1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

precision = precision_score(y_true, y_pred, average="weighted")
recall    = recall_score(y_true, y_pred, average="weighted")
f1        = f1_score(y_true, y_pred, average="weighted")
report    = classification_report(y_true, y_pred)

with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
    f.write(f"Model: {MODEL_TYPE}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n\n")
    f.write(report)

print(f"\n✅ Done! Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}")
print(f"All results saved in '{RESULTS_DIR}/' folder.")