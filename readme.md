# 🎤 Speech Emotion Recognition (SER) Web App

A deep learning–based Speech Emotion Recognition system built using PyTorch and deployed as an interactive Streamlit web application. The system predicts human emotions from speech audio using advanced audio features and a CNN+LSTM hybrid architecture.

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

| Model         | Accuracy | F1-Score | Features Used | Architecture |
|---------------|----------|----------|---------------|--------------|
| **CNN+LSTM**  | **87.5%**| **0.90** | MFCC + Δ + ΔΔ + Mel + Chroma + ZCR + RMS (174 features) | CNN → Bi-LSTM → FC |
| LSTM          | ~68%     | ~0.68    | MFCC only (40 features) | Bi-LSTM → FC |
| CNN           | 67%      | 0.67     | MFCC only (40 features) | Conv1D → FC |
| GRU           | ~66%     | ~0.66    | MFCC only (40 features) | Bi-GRU → FC |
| RNN           | ~65%     | ~0.65    | MFCC only (40 features) | Bi-RNN → FC |
| Random Forest | ~62%     | ~0.62    | MFCC only (40 features) | Classical ML |
| SVM           | ~60%     | ~0.60    | MFCC only (40 features) | Classical ML |
| KNN           | ~58%     | ~0.58    | MFCC only (40 features) | Classical ML |

### 🏆 Winner: CNN+LSTM

**20%+ improvement** over baseline CNN through:
- Richer audio features (174 features vs 40 MFCCs)
- Temporal modeling with bidirectional LSTM
- Batch normalization and dropout for regularization
- Advanced feature engineering (delta MFCCs, mel spectrograms, chroma features)

---

## 🏗️ Project Structure
```
speech_emotion/
│
├── web_app.py                    # Streamlit web app
├── train.py                      # Training script
├── predict.py                    # CLI prediction
├── predict_utils.py              # Prediction utilities
│
├── models/                       # Model architectures
│   ├── cnn.py
│   ├── lstm.py
│   ├── gru.py
│   └── rnn.py
│
├── cnn_lstm_model.py             # Best performing model
├── features.py                   # Feature extraction
├── augment.py                    # Data augmentation
├── dataset.py                    # Dataset loader
│
├── tests/                        # Unit tests
│   ├── test_features.py
│   └── test_model.py
│
├── results_cnn_lstm/             # Best model results
│   └── best_model.pth
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

### Dataset
**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors
- ~1,440 audio files
- Speaker-independent train/test split

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

### Best Model Performance (CNN+LSTM)
- **Accuracy:** 87.5%
- **Precision:** 89.96%
- **Recall:** 89.61%
- **F1-score:** 0.8956
- **Training time:** ~60 epochs (early stopped at 62)

All evaluation plots (loss curves, confusion matrix, ROC-AUC) are saved in `results_cnn_lstm/`.

---

## 📊 Evaluation Metrics

The following metrics were used for all models:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix
- ROC–AUC Curve
- Classification Report

Results are saved in model-specific folders (`results/`, `results_cnn_lstm/`, etc.).

---

## ⚠️ Known Limitations

### Live Recording Accuracy
- The model was trained on **professionally acted emotions** (RAVDESS dataset), so it performs best on exaggerated, theatrical speech
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
- Dataset not included due to size (download RAVDESS separately)
- Trained model weights provided in `results_cnn_lstm/`
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
  title = {Speech Emotion Recognition using CNN+LSTM},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/deepak-kalyani/speech_emotion}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright © 2026 Deepak Kalyani. All rights reserved.**

---

## 🙏 Acknowledgments

- **RAVDESS Dataset** creators for providing high-quality emotional speech data
- **PyTorch** and **Streamlit** communities for excellent frameworks
- **Librosa** developers for powerful audio processing capabilities
- All contributors and researchers in the speech emotion recognition field

---

## 🔥 Project Highlights

✅ Complete ML pipeline (training, evaluation, inference, deployment)  
✅ Multiple model architectures benchmarked  
✅ Advanced feature engineering  
✅ Clean, modular, production-ready code  
✅ Unit tests included  
✅ Interactive web interface  
✅ 87.5% accuracy on speaker-independent test set  
✅ Real-time voice recording capability

---

**⭐ If you found this project helpful, please consider giving it a star on GitHub!**