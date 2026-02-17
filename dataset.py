import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from features import extract_features, extract_features_from_array
from augment import augment_audio

class RAVDESSDataset(Dataset):
    def __init__(self, dataset_path, augment=False):
        self.features = []
        self.labels = []

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    path = os.path.join(root, file)
                    label = int(file[6:8]) - 1

                    # Original
                    self.features.append(extract_features(path))
                    self.labels.append(label)

                    # Augmented versions (training only)
                    if augment:
                        y, sr = librosa.load(path, sr=22050, res_type='kaiser_fast')
                        for aug_y in augment_audio(y, sr):
                            self.features.append(extract_features_from_array(aug_y, sr))
                            self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label    = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label