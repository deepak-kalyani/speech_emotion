import numpy as np
import librosa

def augment_audio(y, sr):
    augmented = []

    # 1. Time stretching
    for rate in [0.9, 1.1]:
        augmented.append(librosa.effects.time_stretch(y, rate=rate))

    # 2. Pitch shifting
    for steps in [-2, 2]:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))

    # 3. Add noise
    augmented.append(y + 0.005 * np.random.randn(len(y)))

    # 4. Time shift
    augmented.append(np.roll(y, int(sr * 0.1)))

    return augmented  # 6 variants per audio file