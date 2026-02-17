import librosa
import numpy as np

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=128):
    y, sr = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    return _build_features(y, sr, n_mfcc, max_len)

def extract_features_from_array(y, sr=22050, n_mfcc=40, max_len=128):
    return _build_features(y, sr, n_mfcc, max_len)

def _build_features(y, sr, n_mfcc=40, max_len=128):
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mel         = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)

    combined = np.vstack([mfcc, delta_mfcc, delta2_mfcc, mel, chroma, zcr, rms])  # (174, time)

    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0, 0), (0, max_len - combined.shape[1])))
    else:
        combined = combined[:, :max_len]

    return combined  # shape: (174, 128)