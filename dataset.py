import os
import torch
from torch.utils.data import Dataset
from features import extract_mfcc

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
        mfcc = extract_mfcc(self.files[idx])      # shape: (40,)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        # shape becomes: (1, 40)  -> correct for Conv1D

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label
