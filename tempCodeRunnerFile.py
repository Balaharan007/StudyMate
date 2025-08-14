import cv2
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import os
from tensorflow.keras.models import load_model
from textblob import TextBlob
import speech_recognition as sr
import matplotlib.pyplot as plt

# Load Pretrained Facial Expression Model
face_emotion_model = load_model(r"D:\IEEE\SoftSkills\fer_model.h5")

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to capture face and predict emotions
def analyze_facial_expression():
    cap = cv2.VideoCapture(1)  # Open webcam

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return None

    print("🎥 Capturing Face... (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=0) / 255.0

            # Predict Emotion
            predictions = face_emotion_model.predict(roi_gray)
            max_index = np.argmax(predictions[0])
            emotion = emotion_labels[max_index]

            # Draw Rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Facial Expression Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return emotion  # Return detected emotion

# Function to record audio for voice analysis
def record_audio(duration=5, samplerate=22050):
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    wav.write("recorded_audio.wav", samplerate, (audio * 32767).astype(np.int16))
    return "recorded_audio.wav"

# Function to analyze voice tone (pitch & intensity)
def analyze_voice_tone(audio_path):
    y, sr = librosa.load(audio_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)  # Extract pitch
    intensity = np.abs(y).mean()  # Compute intensity

    # Visualize Pitch
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Voice Waveform")
    plt.show()

    avg_pitch = np.nanmean(pitch)  # Compute average pitch

    print(f"🎼 Average Pitch: {avg_pitch:.2f} Hz")
    print(f"🔊 Intensity Level: {intensity:.2f}")

    return avg_pitch, intensity

# Function to analyze speech sentiment
def analyze_speech_sentiment(audio_path):
    recognizer = sr.Recognizer()
    audio_text = ""

    with sr.AudioFile(audio_path) as source:
        print("🗣 Converting Speech to Text...")
        audio = recognizer.record(source)
        try:
            audio_text = recognizer.recognize_google(audio)
            print(f"📜 Transcribed Text: {audio_text}")
        except:
            print("❌ Could not recognize speech.")
            return "Neutral"

    # Analyze sentiment
    sentiment_score = TextBlob(audio_text).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    print(f"📝 Sentiment Score: {sentiment_score:.2f} ({sentiment})")
    return sentiment

# Function to calculate Soft Skill Score
def calculate_soft_skill_score(emotion, pitch, intensity, sentiment):
    # Scoring system (adjust based on experiments)
    emotion_score = {"Happy": 9, "Neutral": 7, "Surprise": 6, "Sad": 4, "Angry": 3, "Fear": 2, "Disgust": 1}.get(emotion, 5)
    sentiment_score = {"Positive": 9, "Neutral": 6, "Negative": 3}.get(sentiment, 5)
    
    pitch_score = 10 if 100 <= pitch <= 250 else 5  # Ideal human pitch range
    intensity_score = 10 if intensity > 0.02 else 5  # Loudness threshold

    # Weighted Score Calculation
    final_score = (emotion_score * 0.3) + (sentiment_score * 0.3) + (pitch_score * 0.2) + (intensity_score * 0.2)
    
    print("\n🎯 Soft Skills Score Analysis:")
    print(f"🟢 Emotion Score: {emotion_score}/10")
    print(f"🟢 Sentiment Score: {sentiment_score}/10")
    print(f"🟢 Pitch Score: {pitch_score}/10")
    print(f"🟢 Intensity Score: {intensity_score}/10")
    print(f"⭐ Final Soft Skills Score: {final_score:.2f}/10")
    
    return final_score

# Main Execution
if __name__ == "__main__":
    print("\n🚀 Starting Soft Skills Analysis...\n")

    # Step 1: Facial Expression Analysis
    emotion = analyze_facial_expression()

    # Step 2: Voice & Speech Analysis
    audio_path = record_audio()
    pitch, intensity = analyze_voice_tone(audio_path)
    sentiment = analyze_speech_sentiment(audio_path)

    # Step 3: Calculate Soft Skill Score
    final_score = calculate_soft_skill_score(emotion, pitch, intensity, sentiment)

    print("\n✅ Soft Skills Analysis Complete!")
