import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN block — extracts local patterns from features
        self.conv1 = nn.Conv1d(174, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64,  kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(256)
        self.bn2     = nn.BatchNorm1d(128)
        self.bn3     = nn.BatchNorm1d(64)

        # LSTM block — captures how emotion evolves over time
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True   # looks forward AND backward in time
        )

        # Classifier
        self.fc1 = nn.Linear(128 * 2, 64)  # *2 because bidirectional
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        # x: (batch, 174, 128)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))  # (batch, 256, 128)
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))  # (batch, 128, 128)
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))  # (batch, 64,  128)

        x = x.permute(0, 2, 1)        # (batch, 128, 64) — time first for LSTM
        x, _ = self.lstm(x)           # (batch, 128, 256)
        x = x[:, -1, :]               # take last timestep (batch, 256)

        x = self.relu(self.fc1(x))    # (batch, 64)
        x = self.dropout(x)
        return self.fc2(x)            # (batch, 8)