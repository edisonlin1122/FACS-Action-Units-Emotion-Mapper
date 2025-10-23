#Run using: streamlit run emotionMapperApp.py
# This is the fiile that runs the interactive Streamlit web app

# Imports modules
# streamlit for web app interface
# cv2 for video processing
# mediapipe for facial landmark detection
# tempfile for temporary file handling
# pandas and altair for data handling and visualization

import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import pandas as pd
import altair as alt


# Function to detect specific Action Units (AUs) based on facial landmarks
def detectAU1(landmarks):
    brow = landmarks[65].y
    eye = landmarks[159].y
    return brow - eye < -0.02

def detectAU4(landmarks):
    brow = landmarks[65].y
    innerEye = landmarks[386].y
    return brow - innerEye > 0.02

def detectAU12(landmarks):
    leftCorner = landmarks[61].x
    rightCorner = landmarks[291].x
    mouthWidth = abs(rightCorner - leftCorner)
    return mouthWidth > 0.35

def detectAU15(landmarks):
    corner = landmarks[61].y
    chin = landmarks[17].y
    return corner - chin > 0.05

def detectAU26(landmarks):
    topLip = landmarks[13].y
    bottomLip = landmarks[14].y
    return bottomLip - topLip > 0.03

def mapEmotion(landmarks):
    aus = []
    if detectAU1(landmarks): aus.append("AU1")
    if detectAU4(landmarks): aus.append("AU4")
    if detectAU12(landmarks): aus.append("AU12")
    if detectAU15(landmarks): aus.append("AU15")
    if detectAU26(landmarks): aus.append("AU26")

    if "AU1" in aus and "AU26" in aus:
        emotion = "Surprise"
    elif "AU1" in aus and "AU4" in aus and "AU15" in aus:
        emotion = "Sadness"
    elif "AU4" in aus and "AU26" in aus:
        emotion = "Anger"
    elif "AU12" in aus and len(aus) == 1:
        emotion = "Disgust"
    elif "AU12" in aus:
        emotion = "Happiness"
    else:
        emotion = "Neutral"

    return aus, emotion

# Streamlit app interface
st.set_page_config(page_title="FACS Action Units Emotion Mapper", layout="wide")
st.title("FACS Action Units Emotion Mapper")
st.caption("Import a video file to run facial analysis using FACS Action Units and map expressions to emotions")

# File uploader for video input
uploadedFile = st.file_uploader(
    label="Select a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"]
)

# Stored as temp file for OpenCV processing
if uploadedFile:
    tempFile = tempfile.NamedTemporaryFile(delete=False)
    tempFile.write(uploadedFile.read())
    videoPath = tempFile.name

    st.info("Processing the input...")

# Using MediaPipe's FaceMesh model to track facial landmarks
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(videoPath)
    frameCount = 0
    emotionLog = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(rgbFrame)

# Detect AUs and map to emotions for each frame
        if results.multi_face_landmarks:
            for faceLandmarks in results.multi_face_landmarks:
                aus, emotion = mapEmotion(faceLandmarks.landmark)
                emotionLog.append((frameCount, emotion, ", ".join(aus)))

        frameCount += 1

    cap.release()

    st.success("Processing complete")

# Makes a Dataframe and also counts for emotion frequency
    df = pd.DataFrame(emotionLog, columns=["Frame", "Emotion", "Detected AUs"])
    emotionCounts = df["Emotion"].value_counts().reset_index()
    emotionCounts.columns = ["Emotion", "Count"]

# Layout with video preview
    leftCol, rightCol = st.columns(2)

    with leftCol:
        st.subheader("Video Preview")
        st.video(uploadedFile)

# Layout with DataFrame and Emotion Frequency Chart
    with rightCol:
        st.subheader("Action Units Mapping Frames")
        st.dataframe(df, height=400)

        st.subheader("Emotion Mapping Frequency")
        st.markdown(f"**Total frames analyzed:** {len(df)}")

        chart = alt.Chart(emotionCounts).mark_bar().encode(
            x="Emotion",
            y="Count",
            color="Emotion"
        ).properties(title="Emotion Mapping Frequency")

        st.altair_chart(chart, use_container_width=True)

# Reference section for AU to Emotion mapping
st.markdown("---")
st.subheader("Action Units to Emotion Mapping Reference")
st.markdown("""
This is a reference chart for how the specific Action Units (AUs) are interpreted as emotions:

| Emotion     | Key Action Units |
|-------------|------------------|
| Surprise    | AU1 (Inner Brow Raise), AU26 (Jaw Drop) |
| Sadness     | AU1 (Inner Brow Raise), AU4 (Brow Lower), AU15 (Lip Corner Depressor) |
| Anger       | AU4 (Brow Lower), AU26 (Jaw Drop) |
| Happiness   | AU12 (Lip Corner Pull) |
| Disgust     | AU12 (Lip Corner Pull) — isolated |
| Neutral     | No detected AU1, AU4, AU12, AU15, or AU26 |

MediaPipe landmarks were used to detect facial movements and map them to emotions
""")