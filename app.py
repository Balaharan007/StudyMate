import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from textblob import TextBlob
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

# Retrieve values from .env
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 22050))
AUDIO_DURATION = int(os.getenv("AUDIO_DURATION", 5))
HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH")

# Load Pretrained Facial Expression Model
face_emotion_model = load_model(FER_MODEL_PATH)

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to capture face and predict emotions
def analyze_facial_expression():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return None

    print("🎥 Capturing Face... (Press 'q' to quit)")
    detected_emotion = "Neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.repeat(roi_gray, 3, axis=-1)
            roi_gray = roi_gray / 255.0

            predictions = face_emotion_model.predict(roi_gray)
            max_index = np.argmax(predictions[0])
            detected_emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Facial Expression Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return detected_emotion

def record_audio(duration=AUDIO_DURATION, samplerate=AUDIO_SAMPLE_RATE):
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    wav.write("recorded_audio.wav", samplerate, (audio * 32767).astype(np.int16))
    return "recorded_audio.wav"

def analyze_voice_tone(audio_path):
    y, sr = librosa.load(audio_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    intensity = np.abs(y).mean()

    avg_pitch = np.nanmean(pitch)

    print(f"🎼 Average Pitch: {avg_pitch:.2f} Hz")
    print(f"🔊 Intensity Level: {intensity:.2f}")

    return avg_pitch, intensity

def analyze_speech_sentiment(audio_path):
    recognizer = sr.Recognizer()
    audio_text = ""

    with sr.AudioFile(audio_path) as source:
        print("🗣 Converting Speech to Text...")
        audio = recognizer.record(source)
        try:
            audio_text = recognizer.recognize_google(audio)
            print(f"📜 Transcribed Text: {audio_text}")
        except sr.UnknownValueError:
            print("❌ Could not understand speech.")
            return "Neutral"
        except sr.RequestError:
            print("❌ Speech Recognition service unavailable.")
            return "Neutral"

    sentiment_score = TextBlob(audio_text).sentiment.polarity
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    print(f"📝 Sentiment Score: {sentiment_score:.2f} ({sentiment})")
    return sentiment

def calculate_soft_skill_score(emotion, pitch, intensity, sentiment):
    emotion_score = {"Happy": 9, "Neutral": 7, "Surprise": 6, "Sad": 4, "Angry": 3, "Fear": 2, "Disgust": 1}.get(emotion, 5)
    sentiment_score = {"Positive": 9, "Neutral": 6, "Negative": 3}.get(sentiment, 5)

    pitch_score = 10 if 100 <= pitch <= 250 else 5
    intensity_score = 10 if intensity > 0.02 else 5

    final_score = (emotion_score * 0.3) + (sentiment_score * 0.3) + (pitch_score * 0.2) + (intensity_score * 0.2)

    print("\n🎯 Soft Skills Score Analysis:")
    print(f"🟢 Emotion Score: {emotion_score}/10")
    print(f"🟢 Sentiment Score: {sentiment_score}/10")
    print(f"🟢 Pitch Score: {pitch_score}/10")
    print(f"🟢 Intensity Score: {intensity_score}/10")
    print(f"⭐ Final Soft Skills Score: {final_score:.2f}/10")

    return final_score

def get_improvement_suggestions(score):
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    prompt = {
        "contents": [{
            "parts": [{
                "text": f"A user has a soft skills score of {score}/10. Suggest improvements in communication, confidence, and emotional expression."
            }]
        }]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{endpoint}?key={GEMINI_API_KEY}", json=prompt, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        return "❌ Error: Unable to fetch suggestions."

if __name__ == "__main__":
    print("\n🚀 Starting Soft Skills Analysis...\n")

    emotion = analyze_facial_expression() or "Neutral"
    audio_path = record_audio()
    pitch, intensity = analyze_voice_tone(audio_path)
    sentiment = analyze_speech_sentiment(audio_path)

    final_score = calculate_soft_skill_score(emotion, pitch, intensity, sentiment)

    suggestions = get_improvement_suggestions(final_score)
    print("\n💡 Improvement Suggestions:\n", suggestions)

    print("\n✅ Soft Skills Analysis Complete!")
