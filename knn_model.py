import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from features import extract_mfcc

# -----------------------------
# DATASET PATH
# -----------------------------
DATASET_PATH = "data/RAVDESS"

X_train, y_train = [], []
X_test, y_test = [], []

# -----------------------------
# SPEAKER-INDEPENDENT SPLIT
# -----------------------------
TRAIN_ACTORS = [f"Actor_{i:02d}" for i in range(1, 19)]
TEST_ACTORS  = [f"Actor_{i:02d}" for i in range(19, 25)]

print("🎙️ Training Actors:", TRAIN_ACTORS)
print("🎧 Testing Actors :", TEST_ACTORS)

# -----------------------------
# LOAD DATA
# -----------------------------
print("\n🔄 Extracting MFCC features...")

for actor in TRAIN_ACTORS:
    actor_path = os.path.join(DATASET_PATH, actor)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            mfcc = extract_mfcc(file_path)
            label = int(file[6:8]) - 1

            X_train.append(mfcc)
            y_train.append(label)

for actor in TEST_ACTORS:
    actor_path = os.path.join(DATASET_PATH, actor)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            mfcc = extract_mfcc(file_path)
            label = int(file[6:8]) - 1

            X_test.append(mfcc)
            y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test  = np.array(X_test)
y_test  = np.array(y_test)

print("✅ Feature extraction done")
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -----------------------------
# FEATURE SCALING (MANDATORY)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# KNN MODEL
# -----------------------------
print("\n🚀 Training KNN model...")

knn_model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="euclidean"
)

knn_model.fit(X_train, y_train)

print("✅ KNN training completed")

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n📊 KNN PERFORMANCE (Speaker-Independent)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs("results_knn_si", exist_ok=True)

with open("results_knn_si/metrics_knn_si.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - KNN (Speaker Independent)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results_knn_si/confusion_matrix_knn_si.png")
plt.close()

print("\n✅ KNN results saved in 'results_knn_si/' folder")
