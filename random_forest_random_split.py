import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

X, y = [], []

# -----------------------------
# LOAD FULL DATASET (NO ACTOR SPLIT)
# -----------------------------
print("🔄 Extracting MFCC features (random split)...")

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            mfcc = extract_mfcc(file_path)
            label = int(file[6:8]) - 1

            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", X.shape[0])

# -----------------------------
# RANDOM TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------
print("\n🌲 Training Random Forest (random split)...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("✅ Training completed")

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n📊 RANDOM FOREST (RANDOM SPLIT)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs("results_rf_random", exist_ok=True)

with open("results_rf_random/metrics_rf_random.txt", "w") as f:
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
plt.title("Confusion Matrix - Random Forest (Random Split)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results_rf_random/confusion_matrix_rf_random.png")
plt.close()

print("\n✅ Random-split RF results saved in 'results_rf_random/'")
