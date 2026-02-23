"""
Preprocesses TESS dataset to match RAVDESS format
Creates a unified directory structure for training
"""

import os
import shutil
from pathlib import Path

# Emotion mapping (TESS → RAVDESS label)
EMOTION_MAP = {
    'angry': 4,      # Angry
    'disgust': 6,    # Disgust
    'fear': 5,       # Fearful
    'happy': 2,      # Happy
    'ps': 7,         # Pleasant Surprise → Surprised
    'sad': 3,        # Sad
    'neutral': 0     # Neutral
}

def organize_tess(tess_root="TESS", output_root="data/TESS_processed"):
    """
    Organize TESS files into emotion-based folders
    Files are named like: YAF_youth_fear.wav or OAF_back_angry.wav
    """
    
    os.makedirs(output_root, exist_ok=True)
    
    # Create emotion folders
    for emotion_name, label in EMOTION_MAP.items():
        emotion_folder = os.path.join(output_root, f"emotion_{label}")
        os.makedirs(emotion_folder, exist_ok=True)
    
    total_files = 0
    emotion_counts = {label: 0 for label in EMOTION_MAP.values()}
    
    # Process all WAV files in TESS folder
    for file in os.listdir(tess_root):
        if not file.endswith('.wav'):
            continue
        
        # Extract emotion from filename
        # Format: YAF_word_emotion.wav or OAF_word_emotion.wav
        parts = file.replace('.wav', '').split('_')
        
        if len(parts) < 3:
            print(f"⚠️  Skipping invalid filename: {file}")
            continue
        
        emotion_key = parts[-1].lower()  # Last part is the emotion
        
        if emotion_key not in EMOTION_MAP:
            print(f"⚠️  Unknown emotion in {file}: {emotion_key}")
            continue
        
        label = EMOTION_MAP[emotion_key]
        dest_folder = os.path.join(output_root, f"emotion_{label}")
        
        # Copy file
        src = os.path.join(tess_root, file)
        dst = os.path.join(dest_folder, file)
        shutil.copy2(src, dst)
        
        total_files += 1
        emotion_counts[label] += 1
    
    print(f"\n✅ Processed {total_files} TESS files")
    
    # Show distribution
    print("\n📊 TESS Emotion Distribution:")
    emotion_names = {v: k for k, v in EMOTION_MAP.items()}
    for label in sorted(emotion_counts.keys()):
        count = emotion_counts[label]
        emotion_name = emotion_names.get(label, "unknown")
        print(f"  Emotion {label} ({emotion_name}): {count} files")
    
    return total_files

if __name__ == "__main__":
    print("="*60)
    print("TESS Dataset Preprocessing")
    print("="*60)
    
    if not os.path.exists("TESS"):
        print("\n❌ Error: TESS folder not found!")
        print("Please download TESS and extract it to a folder named 'TESS'")
        exit(1)
    
    # Check if TESS has files
    wav_files = [f for f in os.listdir("TESS") if f.endswith('.wav')]
    print(f"\nFound {len(wav_files)} WAV files in TESS folder")
    
    if len(wav_files) == 0:
        print("❌ No WAV files found in TESS folder!")
        exit(1)
    
    organize_tess()
    print("\n✅ TESS preprocessing complete!")
    print("Files saved to: data/TESS_processed/")