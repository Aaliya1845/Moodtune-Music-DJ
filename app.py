
import streamlit as st
import pandas as pd
from transformers import pipeline
from deepface import DeepFace
import speech_recognition as sr
from pydub import AudioSegment
import os

# Set page configuration
st.set_page_config(layout="wide")

# --- Load Data and Models ---

# Songs DataFrame
songs_data = [
    ["happy", "Hindi", "Romantic", "Kesariya", "Arijit Singh", "https://youtu.be/BddP6PYo2gs"],
    ["happy", "Hindi", "Romantic", "Raatan Lambiyan", "Jubin Nautiyal", "https://youtu.be/gvyUuxdRdR4"],
    ["happy", "Hindi", "Feel Good", "Ilahi", "Arijit Singh", "https://youtu.be/fdubeMFwuGs"],
    ["happy", "Hindi", "Party", "Kar Gayi Chull", "Badshah", "https://youtu.be/NTHz9ephYTw"],
    ["happy", "Hindi", "Motivational", "Kar Har Maidaan Fateh", "Sukhwinder Singh", "https://youtu.be/KJxS6XHkYlQ"],
    ["happy", "Punjabi", "Pop", "Lover", "Diljit Dosanjh", "https://youtu.be/DP8TlBLPc8A"],
    ["happy", "Punjabi", "Pop", "GOAT", "Diljit Dosanjh", "https://youtu.be/cl0a3i2wFcc"],
    ["happy", "English", "Pop", "Happy", "Pharrell Williams", "https://youtu.be/ZbZSe6N_BXs"],
    ["happy", "Punjabi", "Party", "Laembadgini", "Diljit Dosanjh", "https://youtu.be/placeholder5"],

    ["sad", "Hindi", "Emotional", "Channa Mereya", "Arijit Singh", "https://youtu.be/284Ov7ysmfA"],
    ["sad", "Hindi", "Emotional", "Agar Tum Saath Ho", "Alka Yagnik", "https://youtu.be/sK7riqg2mr4"],
    ["sad", "Hindi", "Emotional", "Tujhe Bhula Diya", "Mohit Chauhan", "https://youtu.be/vM0xQkG3N2I"],
    ["sad", "Hindi", "Soft", "Khairiyat", "Arijit Singh", "https://youtu.be/hoNb6HuNmU0"],
    ["sad", "Hindi", "Emotional", "Phir Bhi Tumko Chaahunga", "Arijit Singh", "https://youtu.be/_iktURk0X-A"],
    ["sad", "Punjabi", "Emotional", "Qismat", "Ammy Virk", "https://youtu.be/jF2fztq2YcQ"],
    ["sad", "Punjabi", "Emotional", "Soch", "Harrdy Sandhu", "https://youtu.be/rRr1qiJRsXk"],
    ["sad", "English", "Emotional", "Someone Like You", "Adele", "https://youtu.be/hLQl3WQQoQ0"],

    ["angry", "Hindi", "Rock", "Zinda", "Siddharth Mahadevan", "https://youtu.be/Awkx1jXgGZE"],
    ["angry", "Punjabi", "Party", "Born To Shine", "Diljit Dosanjh", "https://youtu.be/dCmp56tSSmA"],
    ["angry", "English", "Rock", "Believer", "Imagine Dragons", "https://youtu.be/7wtfhZwyrcc"],

    ["neutral", "Hindi", "Romantic", "Raabta", "Arijit Singh", "https://youtu.be/zlt38OOqwDc"],
    ["neutral", "Hindi", "Soft", "Apna Bana Le", "Arijit Singh", "https://youtu.be/ElZfdU54Cp8"],
    ["neutral", "Hindi", "Soft", "Tum Se Hi", "Mohit Chauhan", "https://youtu.be/mt9xg0mmt28"],
    ["neutral", "Hindi", "Romantic", "Hawayein", "Arijit Singh", "https://youtu.be/cYOB941gyXI"],
    ["neutral", "Punjabi", "Pop", "Excuses", "AP Dhillon", "https://youtu.be/vX2cDW8LUWk"],
    ["neutral", "English", "Romantic", "Perfect", "Ed Sheeran", "https://youtu.be/2Vv-BfVoq4g"]
]

df = pd.DataFrame(
    songs_data,
    columns=["emotion", "language", "genre", "song", "artist", "youtube"]
)

# Emotion Classifier (Hugging Face Transformers)
@st.cache_resource
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )
classifier = load_emotion_classifier()

# Emotion Mapping
emotion_map = {
    "joy": "happy", "love": "happy", "sadness": "sad", "anger": "angry",
    "fear": "sad", "surprise": "happy", "neutral": "neutral",
    "happy": "happy", "sad": "sad", "angry": "angry", "disgust": "angry",
    "contempt": "angry", "surprise":"happy", "fear": "sad", "anxiety": "sad"
}

# --- Recommendation Function ---
def recommend_songs(emotion, language="All", genre="All", n=5):
    rec = df.copy()

    rec = rec[
        rec["emotion"].str.lower() == emotion.lower()
    ]

    if language != "All":
        rec = rec[
            rec["language"].str.lower() == language.lower()
        ]

    if genre != "All":
        rec = rec[
            rec["genre"].str.lower() == genre.lower()
        ]

    if len(rec) == 0:
        st.write("No songs found for your criteria.")
        return

    rec = rec.sample(
        min(n, len(rec))
    )

    st.subheader(f"Recommended Songs for {emotion.capitalize()} Mood:")
    for _, row in rec.iterrows():
        st.markdown(f"### ðŸŽµ {row['song']} - {row['artist']}")
        st.write(f"Language: {row['language']}")
        st.write(f"Genre: {row['genre']}")
        st.markdown(f"[Listen on YouTube]({row['youtube']})")
        st.markdown("--- Say that you love me. --- ")

# --- Streamlit UI ---
st.title("ðŸŽ¶ Emotion-Based Song Recommender")
st.markdown("Upload an image, audio, or enter text to get song recommendations based on your detected emotion.")

# Emotion detection options
option = st.sidebar.radio(
    "Choose your emotion input method:",
    ("Text Input", "Image Upload", "Audio Upload")
)

predicted_emotion = "neutral"

with st.spinner('Processing...'):
    if option == "Text Input":
        text_input = st.text_area("Tell me how you are feeling:", "I am feeling very happy today!")
        if st.button("Analyze Text Emotion") and text_input:
            result = classifier(text_input)
            pred = result[0][0]["label"].lower()
            predicted_emotion = emotion_map.get(pred, "neutral")
            st.success(f"Detected Emotion: {predicted_emotion.capitalize()}")

    elif option == "Image Upload":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image Emotion"):
                try:
                    result = DeepFace.analyze(
                        img_path="temp_image.jpg",
                        actions=["emotion"],
                        enforce_detection=False
                    )
                    dominant_emotion = result[0]["dominant_emotion"]
                    predicted_emotion = emotion_map.get(dominant_emotion, "neutral")
                    st.success(f"Detected Emotion: {predicted_emotion.capitalize()}")
                except Exception as e:
                    st.error(f"Error analyzing image: {e}. Make sure a face is clearly visible.")
                finally:
                    if os.path.exists("temp_image.jpg"):
                        os.remove("temp_image.jpg")

    elif option == "Audio Upload":
        uploaded_audio = st.file_uploader("Upload an audio file (m4a, wav)", type=["m4a", "wav"])
        if uploaded_audio:
            audio_path = "temp_audio." + uploaded_audio.type.split('/')[-1]
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            st.audio(uploaded_audio, format=uploaded_audio.type)

            if st.button("Analyze Audio Emotion"):
                try:
                    # Convert to WAV if m4a
                    if audio_path.endswith(".m4a"):
                        wav_path = "temp_audio.wav"
                        AudioSegment.from_file(audio_path, format="m4a").export(wav_path, format="wav")
                        current_audio_path = wav_path
                    else:
                        current_audio_path = audio_path

                    r = sr.Recognizer()
                    with sr.AudioFile(current_audio_path) as source:
                        audio = r.record(source)
                        text = r.recognize_google(audio)
                        st.info(f"Recognized Text: {text}")
                        result = classifier(text)
                        pred = result[0][0]["label"].lower()
                        predicted_emotion = emotion_map.get(pred, "neutral")
                        st.success(f"Detected Emotion: {predicted_emotion.capitalize()}")
                except sr.UnknownValueError:
                    st.error("Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                finally:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    if 'wav_path' in locals() and os.path.exists(wav_path):
                        os.remove(wav_path)

# --- Song Recommendation Filters ---
st.sidebar.markdown("## Filter Recommendations")
language_choice = st.sidebar.selectbox(
    "Select Language:",
    ("All", "Hindi", "Punjabi", "English"),
    index=0
)
genre_choice = st.sidebar.selectbox(
    "Select Genre:",
    ("All", "Romantic", "Party", "Pop", "Soft", "Emotional", "Rock", "Feel Good", "Motivational"),
    index=0
)

if predicted_emotion:
    recommend_songs(
        emotion=predicted_emotion,
        language=language_choice,
        genre=genre_choice
    )
