"""
Combined dataset loader for RAVDESS + TESS
"""

import os
import torch
from torch.utils.data import Dataset
from features import extract_features

class CombinedEmotionDataset(Dataset):
    def __init__(self, ravdess_path="data/RAVDESS", tess_path="data/TESS_processed", augment=False):
        self.features = []
        self.labels = []
        
        print("Loading datasets...")
        
        # Load RAVDESS
        if os.path.exists(ravdess_path):
            ravdess_count = self._load_ravdess(ravdess_path, augment)
            print(f"  ✅ RAVDESS: {ravdess_count} samples")
        
        # Load TESS
        if os.path.exists(tess_path):
            tess_count = self._load_tess(tess_path)
            print(f"  ✅ TESS: {tess_count} samples")
        
        print(f"  📊 Total: {len(self.features)} samples")
    
    def _load_ravdess(self, path, augment):
        """Load RAVDESS dataset"""
        count = 0
        
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".wav"):
                    filepath = os.path.join(root, file)
                    label = int(file[6:8]) - 1  # RAVDESS emotion encoding
                    
                    # Original
                    self.features.append(extract_features(filepath))
                    self.labels.append(label)
                    count += 1
        
        return count
    
    def _load_tess(self, path):
        """Load TESS dataset"""
        count = 0
        
        for emotion_folder in os.listdir(path):
            folder_path = os.path.join(path, emotion_folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Extract label from folder name (e.g., "emotion_4" → 4)
            try:
                label = int(emotion_folder.split('_')[1])
            except:
                continue
            
            # Skip emotions not in RAVDESS (TESS has no calm=1)
            if label == 1:  # Calm - TESS doesn't have this
                continue
            
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    filepath = os.path.join(folder_path, file)
                    self.features.append(extract_features(filepath))
                    self.labels.append(label)
                    count += 1
        
        return count
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label