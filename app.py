import streamlit as st
import pandas as pd
from transformers import pipeline
from deepface import DeepFace
import speech_recognition as sr
import tempfile

st.set_page_config(page_title="AI Emotion To Music", page_icon="🎵")

@st.cache_data
def load_songs():
    return pd.read_csv("songs.csv")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )

emotion_map={
"joy":"happy","love":"happy","sadness":"sad",
"anger":"angry","fear":"sad","surprise":"surprise",
"neutral":"neutral","happy":"happy","sad":"sad","angry":"angry"
}

def recommend(df, emotion, language="All", genre="All"):
    rec=df[df["emotion"].str.lower()==emotion.lower()]
    if language!="All":
        rec=rec[rec["language"].str.lower()==language.lower()]
    if genre!="All":
        rec=rec[rec["genre"].str.lower()==genre.lower()]
    return rec.sample(min(5,len(rec))) if len(rec) else rec

df=load_songs()
classifier=load_model()

st.title("🎵 AI Emotion To Music Generator")

language=st.sidebar.selectbox("Language",["All","Hindi","Punjabi","English"])
genre=st.sidebar.selectbox("Genre",["All","Romantic","Party","Pop","Soft","Emotional","Motivational","Rock"])

tab1,tab2,tab3=st.tabs(["Text","Image","Voice"])

with tab1:
    text=st.text_area("How are you feeling?")
    if st.button("Detect Text"):
        res=classifier(text)
        emo=emotion_map.get(res[0][0]["label"].lower(),"neutral")
        st.success(f"Emotion: {emo}")
        st.dataframe(recommend(df,emo,language,genre))

with tab2:
    img=st.file_uploader("Upload image",type=["jpg","jpeg","png"])
    if img and st.button("Detect Image"):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as f:
            f.write(img.read())
            p=f.name
        res=DeepFace.analyze(p,actions=["emotion"],enforce_detection=False)
        emo=emotion_map.get(res[0]["dominant_emotion"],"neutral")
        st.success(f"Emotion: {emo}")
        st.dataframe(recommend(df,emo,language,genre))

with tab3:
    audio=st.file_uploader("Upload wav",type=["wav"])
    if audio and st.button("Detect Voice"):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as f:
            f.write(audio.read())
            p=f.name
        r=sr.Recognizer()
        with sr.AudioFile(p) as source:
            a=r.record(source)
        text=r.recognize_google(a)
        st.write("Recognized:",text)
        res=classifier(text)
        emo=emotion_map.get(res[0][0]["label"].lower(),"neutral")
        st.success(f"Emotion: {emo}")
        st.dataframe(recommend(df,emo,language,genre))
