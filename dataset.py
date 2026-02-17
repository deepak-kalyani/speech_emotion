import os
import torch
from torch.utils.data import Dataset
from features import extract_features

class RAVDESSDataset(Dataset):
    def __init__(self, dataset_path):
        self.files = []
        self.labels = []

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    self.files.append(os.path.join(root, file))
                    self.labels.append(int(file[6:8]) - 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = extract_features(self.files[idx])   # shape: (174, 128)
        features = torch.tensor(features, dtype=torch.float32)
        # shape: (174, 128) — 174 channels, 128 time steps
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label