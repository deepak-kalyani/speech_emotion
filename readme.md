# 🎤 Speech Emotion Recognition (SER) Web App

A deep learning–based Speech Emotion Recognition system built using PyTorch and deployed as an interactive Streamlit web application. The system predicts human emotions from speech audio using advanced audio features and a CNN+LSTM hybrid architecture trained on **4,240 samples** from combined RAVDESS and TESS datasets.

---

## 🚀 Features

- 🎵 Emotion prediction from .wav audio files
- 🎙️ Live voice recording and real-time prediction
- 🧠 Multiple deep learning models (CNN, CNN+LSTM, RNN, LSTM, GRU)
- ⚡ GPU acceleration supported (CUDA)
- 📊 Confidence scores and emotion probabilities
- 🌐 Interactive web interface using Streamlit
- 🔬 Rich audio features (MFCCs, deltas, mel spectrograms, chroma, ZCR, RMS)

---

## 🧠 Emotions Supported

The model classifies speech into **8 emotions**:

| Label | Emotion   |
|-------|-----------|
| 0     | Neutral   |
| 1     | Calm      |
| 2     | Happy     |
| 3     | Sad       |
| 4     | Angry     |
| 5     | Fearful   |
| 6     | Disgust   |
| 7     | Surprised |

---

## 📊 Model Comparison

All models evaluated on speaker-independent test split (70% train / 15% val / 15% test).

| Model         | Accuracy | Precision | Recall | F1-Score | Dataset | Features |
|---------------|----------|-----------|--------|----------|---------|----------|
| **CNN+LSTM**  | **94.1%**| **94.1%** | **94.1%** | **94.1%** | **RAVDESS + TESS (4,240 samples)** | **174 features** |
| CNN+LSTM      | 87.5%    | 90.0%     | 89.6%  | 89.6%    | RAVDESS only (1,440 samples) | 174 features |
| LSTM          | ~68%     | ~68%      | ~68%   | ~68%     | RAVDESS only | 40 features |
| CNN           | 67%      | 67%       | 67%    | 67%      | RAVDESS only | 40 features |
| GRU           | ~66%     | ~66%      | ~66%   | ~66%     | RAVDESS only | 40 features |
| RNN           | ~65%     | ~65%      | ~65%   | ~65%     | RAVDESS only | 40 features |
| Random Forest | ~62%     | ~62%      | ~62%   | ~62%     | RAVDESS only | 40 features |
| SVM           | ~60%     | ~60%      | ~60%   | ~60%     | RAVDESS only | 40 features |
| KNN           | ~58%     | ~58%      | ~58%   | ~58%     | RAVDESS only | 40 features |

### 🏆 Final Winner: CNN+LSTM on Combined Dataset

**27% improvement** from baseline CNN through:
- Combined RAVDESS + TESS datasets (3x more training data → 4,240 samples)
- Rich 174-feature audio representation (MFCC + deltas + mel + chroma + ZCR + RMS)
- CNN+LSTM hybrid architecture for temporal modeling
- Batch normalization and dropout regularization

**Per-emotion performance (F1-scores):**
- Surprised: 97%
- Angry: 96%
- Disgust: 96%
- Fearful: 94%
- Neutral: 94%
- Happy: 93%
- Calm: 91%
- Sad: 88%

---

## 🏗️ Project Structure
```
speech_emotion/
│
├── web_app.py                    # Streamlit web app
├── train.py                      # Training script (RAVDESS only)
├── train_combined.py             # Training script (RAVDESS + TESS)
├── prepare_tess.py               # TESS dataset preprocessing
├── predict.py                    # CLI prediction
├── predict_utils.py              # Prediction utilities
│
├── models/                       # Model architectures
│   ├── __init__.py
│   ├── cnn.py
│   ├── lstm.py
│   ├── gru.py
│   └── rnn.py
│
├── cnn_lstm_model.py             # Best performing model
├── features.py                   # Feature extraction (174 features)
├── augment.py                    # Data augmentation
├── dataset.py                    # RAVDESS dataset loader
├── dataset_combined.py           # Combined dataset loader
│
├── tests/                        # Unit tests
│   ├── test_features.py
│   └── test_model.py
│
├── results_combined/             # Best model results
│   └── best_model.pth            # 94.1% accuracy model
│
├── results_cnn_lstm/             # RAVDESS-only results
│   └── best_model.pth            # 87.5% accuracy model
│
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/deepak-kalyani/speech_emotion.git
cd speech_emotion
```

### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Web App
```bash
streamlit run web_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

### App Capabilities:
- Upload `.wav` audio files
- Record live voice for real-time emotion detection
- Compare predictions across multiple models
- View confidence scores and emotion probabilities
- Interactive, real-time predictions

---

## 🧪 Model Training

### Datasets

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- 1,440 audio files
- 8 emotions with 2 intensity levels
- Speaker-independent train/test split

**TESS** (Toronto Emotional Speech Set)
- 2 female actors (ages 26 and 64)
- 2,800 audio files
- 7 emotions (no "calm" emotion)
- High-quality studio recordings

**Combined:** 4,240 total samples for training

### Features Extracted

- **MFCCs** (40 coefficients)
- **Delta MFCCs** (rate of change)
- **Delta-Delta MFCCs** (acceleration)
- **Mel Spectrogram** (40 bands)
- **Chroma Features** (12 pitch classes)
- **Zero-Crossing Rate**
- **RMS Energy**

**Total: 174 features × 128 time steps**

### Training Configuration

- Train/Val/Test: 70% / 15% / 15%
- Early stopping (patience: 10)
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau
- Batch size: 32

### Best Model Performance (CNN+LSTM on RAVDESS + TESS)

- **Accuracy:** 94.1%
- **Precision:** 94.1%
- **Recall:** 94.1%
- **F1-score:** 0.9411
- **Dataset:** 4,240 samples (RAVDESS + TESS combined)
- **Training time:** ~60 epochs (early stopped)

**Detailed per-emotion metrics:**
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Neutral | 0.94 | 0.94 | 0.94 | 87 |
| Calm | 0.91 | 0.91 | 0.91 | 54 |
| Happy | 0.91 | 0.94 | 0.93 | 107 |
| Sad | 0.92 | 0.85 | 0.88 | 106 |
| Angry | 0.96 | 0.97 | 0.96 | 127 |
| Fearful | 0.93 | 0.94 | 0.94 | 122 |
| Disgust | 0.96 | 0.96 | 0.96 | 114 |
| Surprised | 0.96 | 0.99 | 0.97 | 136 |

All evaluation plots (loss curves, confusion matrix, classification report) are saved in `results_combined/`.

---

## 📊 Evaluation Metrics

The following metrics were used for all models:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix
- Classification Report

Results are saved in model-specific folders:
- `results_combined/` — RAVDESS + TESS (94.1% accuracy) ⭐
- `results_cnn_lstm/` — RAVDESS only (87.5% accuracy)
- `results/`, `results_rnn/`, `results_lstm/`, `results_gru/` — Other models

---

## ⚠️ Known Limitations

### Live Recording Accuracy
- The model was trained on **professionally acted emotions** (RAVDESS + TESS datasets), so it performs best on exaggerated, theatrical speech
- Natural conversational speech may not be classified accurately
- **Recommendation:** For live recording, speak with exaggerated emotions and clear enunciation
- **Example:** Instead of saying "how are you?" normally, try shouting angrily or speaking with theatrical sadness

### Future Work
This is a known challenge in speech emotion recognition. Planned improvements:
- Fine-tune on real-world conversational emotional speech
- Collect and label natural emotion dataset
- Add domain adaptation techniques
- Implement noise reduction preprocessing

---

## 🧪 Running Tests
```bash
python tests/test_features.py
python tests/test_model.py
```

---

## 💡 Technologies Used

- **Python 3.10+**
- **PyTorch** (deep learning)
- **Librosa** (audio processing)
- **NumPy** (numerical computing)
- **Scikit-learn** (metrics & preprocessing)
- **Streamlit** (web interface)
- **Matplotlib** (visualization)

---

## 📌 Notes

- GPU is optional — CPU works fine for inference
- Datasets not included due to size (download RAVDESS and TESS separately)
- Trained model weights provided in `results_combined/` (best) and `results_cnn_lstm/`
- Data augmentation available but disabled by default for faster training

---

## 🚀 Deployment

This project is ready for deployment on:
- **Streamlit Cloud** (recommended)
- **Hugging Face Spaces**
- **Heroku**
- Any cloud platform supporting Python web apps

---

## 👨‍💻 Author

**Deepak Kalyani**

[![GitHub](https://img.shields.io/badge/GitHub-deepak--kalyani-181717?style=flat&logo=github)](https://github.com/deepak-kalyani)

*Built as a complete end-to-end ML application using PyTorch and Streamlit*

---

## 📜 Citation

If you use this project in your research or work, please cite:
```bibtex
@software{kalyani2026speech,
  author = {Kalyani, Deepak},
  title = {Speech Emotion Recognition using CNN+LSTM on Combined RAVDESS and TESS Datasets},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/deepak-kalyani/speech_emotion},
  note = {94.1\% accuracy on 4,240 samples}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Deepak Kalyani. All rights reserved.**

---

## 🙏 Acknowledgments

- **RAVDESS Dataset** creators for providing high-quality emotional speech data
- **TESS Dataset** creators (University of Toronto) for emotional speech recordings
- **PyTorch** and **Streamlit** communities for excellent frameworks
- **Librosa** developers for powerful audio processing capabilities
- All contributors and researchers in the speech emotion recognition field

---

## 🔥 Project Highlights

✅ **94.1% accuracy** on combined RAVDESS + TESS dataset (4,240 samples)  
✅ Complete ML pipeline (training, evaluation, inference, deployment)  
✅ Multiple model architectures benchmarked (9 models compared)  
✅ Advanced feature engineering (174 features extracted)  
✅ Clean, modular, production-ready code  
✅ Unit tests included  
✅ Interactive web interface with live recording  
✅ Balanced performance across all 8 emotions (88-97% F1-scores)  
✅ Professional documentation and attribution

---

**⭐ If you found this project helpful, please consider giving it a star on GitHub!**

---

## 📈 Performance Evolution

| Stage | Accuracy | Dataset | Key Improvement |
|-------|----------|---------|-----------------|
| Baseline CNN | 67% | RAVDESS (1,440) | Initial model |
| CNN+LSTM | 87.5% | RAVDESS (1,440) | +20.5% — Hybrid architecture + rich features |
| **CNN+LSTM** | **94.1%** | **RAVDESS + TESS (4,240)** | **+6.6% — More diverse training data** |

**Total improvement: +27.1% from baseline!**