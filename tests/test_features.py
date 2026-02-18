import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from features import extract_features

def test_feature_shape():
    features = extract_features("sample.wav")
    assert features.shape == (174, 128), f"Expected (174, 128), got {features.shape}"

def test_no_nan_in_features():
    features = extract_features("sample.wav")
    assert not np.any(np.isnan(features)), "Features contain NaN values"

if __name__ == "__main__":
    test_feature_shape()
    test_no_nan_in_features()
    print("✅ All feature tests passed!")