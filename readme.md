🎤 Speech Emotion Recognition (SER) Web App

A deep learning–based Speech Emotion Recognition system built using PyTorch and deployed as an interactive Streamlit web application.
The model predicts human emotions from speech audio using MFCC features and a CNN architecture.

🚀 Features

🎵 Emotion prediction from .wav audio files

🎙️ Live voice recording and real-time prediction

🧠 CNN-based deep learning model (PyTorch)

⚡ GPU acceleration supported (CUDA)

📊 Confidence score for predictions

🌐 Interactive web interface using Streamlit

🧠 Emotions Supported

The model classifies speech into 8 emotions:

Label	Emotion
0	Neutral
1	Calm
2	Happy
3	Sad
4	Angry
5	Fearful
6	Disgust
7	Surprised
🏗️ Project Structure
speech_emotion/
│
├── app.py                 # Streamlit web app
├── predict.py             # CLI-based prediction
├── predict_utils.py       # Prediction logic
│
├── model.py               # CNN architecture
├── features.py            # MFCC feature extraction
│
├── results/
│   └── best_emotion_cnn.pth   # Trained model weights
│
├── requirements.txt       # Required libraries
└── README.md              # Project documentation

⚙️ Installation & Setup
1️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Web App (Recommended)
streamlit run web_app.py


Then open the browser URL shown in the terminal.

App Capabilities:

Upload a .wav audio file

Record voice using microphone

Get predicted emotion + confidence

▶️ Run Prediction via Terminal (Optional)

Place a .wav file in the project folder

Update the file path in predict.py

Run:

python predict.py

🧪 Model Training (Already Done)

Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Features: MFCC (Mel-Frequency Cepstral Coefficients)

Architecture: Convolutional Neural Network (CNN)

Train/Validation/Test Split: 70% / 15% / 15%

Early stopping applied to prevent overfitting

Final Performance (Test Set):

Accuracy: ~67%

F1-score: ~0.67

ROC–AUC: ~0.94

📊 Evaluation Metrics

The following metrics were used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC–AUC Curve

(All evaluation results and plots are saved in the results/ folder.)

💡 Technologies Used

Python 3.10

PyTorch

Librosa

NumPy

Scikit-learn

Streamlit

Matplotlib

Sounddevice

📌 Notes

GPU is optional for prediction (CPU works fine).

Dataset files are not included due to size and licensing.

The trained model (.pth) is already provided.

📈 Future Improvements (Optional)

CNN + LSTM architecture

Mel-spectrogram features

Real-time emotion visualization

Deployment on Streamlit Cloud / HuggingFace Spaces

👨‍💻 Author

Speech Emotion Recognition Project
Built as a complete end-to-end ML application using PyTorch and Streamlit.

🔥 This project demonstrates a full ML pipeline — training, evaluation, inference, and deployment.