import streamlit as st
import pandas as pd
import speech_recognition as sr

# Load songs dataset
songs = pd.read_csv("songs.csv")

# Emotion detection using keywords
def detect_emotion(text):

    text = text.lower()

    happy_words = [
        "happy","good","great","awesome",
        "love","excited","joy","fun"
    ]

    sad_words = [
        "sad","cry","depressed",
        "lonely","hurt","bad"
    ]

    angry_words = [
        "angry","mad","hate",
        "annoyed","frustrated"
    ]

    for word in happy_words:

        if word in text:

            return "happy"

    for word in sad_words:

        if word in text:

            return "sad"

    for word in angry_words:

        if word in text:

            return "angry"

    return "neutral"


# Recommend songs
def recommend_song(emotion, language):

    rec = songs.copy()

    rec.columns = rec.columns.str.lower()

    if "emotion" in rec.columns:

        rec = rec[
            rec["emotion"].str.lower()
            ==
            emotion.lower()
        ]

    if language != "All":

        rec = rec[
            rec["language"].str.lower()
            ==
            language.lower()
        ]

    if len(rec)==0:

        return None

    return rec.sample(
        min(5,len(rec))
    )


# App Title

st.title("🎵 MoodTune Music DJ")

st.write(
"AI Emotion to Music Recommendation"
)

language = st.selectbox(

    "Choose Language",

    ["All",
     "Hindi",
     "Punjabi",
     "English"]

)


# TEXT TAB

st.header("Text Emotion")

text = st.text_input(

    "How are you feeling today?"

)

if st.button("Recommend From Text"):

    emotion = detect_emotion(text)

    st.success(

        f"Detected Emotion : {emotion}"

    )

    rec = recommend_song(

        emotion,

        language

    )

    if rec is not None:

        st.dataframe(rec)

    else:

        st.warning(

            "No songs found"

        )


# VOICE TAB

st.header("Voice Emotion")

audio_file = st.file_uploader(

    "Upload WAV Audio",

    type=["wav"]

)

if st.button("Recommend From Voice"):

    if audio_file is not None:

        r = sr.Recognizer()

        with sr.AudioFile(audio_file) as source:

            audio = r.record(source)

        text = r.recognize_google(audio)

        st.write(

            "Recognized Text :",

            text

        )

        emotion = detect_emotion(text)

        st.success(

            f"Detected Emotion : {emotion}"

        )

        rec = recommend_song(

            emotion,

            language

        )

        if rec is not None:

            st.dataframe(rec)

        else:

            st.warning(

                "No songs found"

            )
