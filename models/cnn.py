import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64 * 40, 8)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
