## FACS Action Units Emotion Mapper

This project implements a Facial Action Coding System (FACS)-based pipeline for detecting facial Action Units (AUs) and mapping them to emotions using MediaPipe FaceMesh

It contains an interactive web interface (emotionMapperApp.py) built with Streamlit

---

## Overview

The system analyzes facial movements frame-by-frame from video input

It identifies key Action Units (AUs) — muscle-based facial indicators — and maps specific combinations to emotional expressions like happiness, sadness, surprise, anger, disgust, and neutrality

### Example Mapping

Emotion | Key Action Units
--------|-----------------
Surprise | AU1 (Inner Brow Raise), AU26 (Jaw Drop)
Sadness  | AU1 (Inner Brow Raise), AU4 (Brow Lower), AU15 (Lip Corner Depressor)
Anger    | AU4 (Brow Lower), AU26 (Jaw Drop)
Happiness | AU12 (Lip Corner Pull)
Disgust  | AU12 (Lip Corner Pull) — isolated
Neutral  | No detected AU1, AU4, AU12, AU15, or AU26

---

## Features

- Facial landmark detection using MediaPipe FaceMesh
- Rule-based AU detection from specific landmark movements
- Emotion mapping based on FACS rules
- Real-time OpenCV visualization
- Interactive Streamlit web app
- Automatic emotion frequency analysis and visual charts using Altair
- Frame-by-frame AU logging in a DataFrame

---

## Installation
git clone https://github.com/edisonlin1122/FACS-Action-Units-Emotion-Mapper.git
cd FACS-Action-Units-Emotion-Mapper

Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

---

## Usage

### Web App Mode (Interactive Analysis)

Run the Streamlit app for uploading and analyzing videos:
streamlit run emotionMapperApp.py

Features:

- Upload mp4, mov, avi, mkv, or webm video files
- View the original video and AU-based emotion results
- Inspect a DataFrame of detected emotions per frame
- Explore a bar chart showing emotion frequency distribution