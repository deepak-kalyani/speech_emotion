import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from cnn_lstm_model import CNNLSTMModel

def test_model_output_shape():
    model = CNNLSTMModel()
    dummy = torch.randn(4, 174, 128)
    output = model(dummy)
    assert output.shape == (4, 8), f"Expected (4, 8), got {output.shape}"

def test_model_runs_on_cpu():
    model = CNNLSTMModel()
    dummy = torch.randn(1, 174, 128)
    output = model(dummy)
    assert output is not None

if __name__ == "__main__":
    test_model_output_shape()
    test_model_runs_on_cpu()
    print("✅ All model tests passed!")