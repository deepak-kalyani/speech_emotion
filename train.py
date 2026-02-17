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

# -----------------------------
# SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# DATASET SPLIT
# -----------------------------
dataset = RAVDESSDataset("data/RAVDESS")
total_size = len(dataset)

train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# -----------------------------
# MODEL
# -----------------------------
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# EARLY STOPPING
# -----------------------------
max_epochs = 100
patience = 7
best_val_acc = 0
epochs_no_improve = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# -----------------------------
# TRAINING LOOP
# -----------------------------
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
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    correct, total, val_loss = 0, 0, 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{max_epochs} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_emotion_cnn.pth")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("⏹ Early stopping triggered")
        break

# -----------------------------
# PLOTS: LOSS & ACCURACY
# -----------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
plt.close()

# -----------------------------
# TEST EVALUATION
# -----------------------------
model.load_state_dict(torch.load(f"{RESULTS_DIR}/best_emotion_cnn.pth"))
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(torch.argmax(probs, 1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# -----------------------------
# METRICS
# -----------------------------
precision = precision_score(y_true, y_pred, average="weighted")
recall    = recall_score(y_true, y_pred, average="weighted")
f1        = f1_score(y_true, y_pred, average="weighted")

report = classification_report(y_true, y_pred)

with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()

# -----------------------------
# ROC–AUC (MULTI-CLASS)
# -----------------------------
y_true_bin = label_binarize(y_true, classes=list(range(8)))

fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_prob).ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], "--")
plt.legend()
plt.title("ROC-AUC Curve")
plt.savefig(f"{RESULTS_DIR}/roc_auc.png")
plt.close()

print("✅ Training complete. All results saved in 'results/' folder.")
