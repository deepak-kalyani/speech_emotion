import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(174, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool    = nn.AdaptiveAvgPool1d(16)  # shrinks time to 16
        self.fc1     = nn.Linear(64 * 16, 128)
        self.fc2     = nn.Linear(128, 8)

    def forward(self, x):
        # x shape: (batch, 174, 128)
        x = self.relu(self.conv1(x))   # (batch, 256, 128)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))   # (batch, 128, 128)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))   # (batch, 64, 128)
        x = self.pool(x)               # (batch, 64, 16)
        x = x.view(x.size(0), -1)      # (batch, 1024)
        x = self.relu(self.fc1(x))     # (batch, 128)
        x = self.dropout(x)
        return self.fc2(x)             # (batch, 8)