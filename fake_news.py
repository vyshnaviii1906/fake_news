import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import tempfile

# Set up Streamlit Page Config
st.set_page_config(page_title="AI Detection Suite", layout="wide")

# Load Fake News Model
@st.cache_resource
def load_fake_news_model():
    model_name = "microsoft/deberta-v3-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

fake_news_model, tokenizer = load_fake_news_model()

# Fetch and Analyze News from URL
def fetch_and_analyze_news(news_url):
    try:
        response = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        news_text = " ".join([p.get_text() for p in paragraphs])

        if len(news_text) < 100:
            return "Error: Could not extract enough text from URL.", None, None

        # Analyze the extracted news text
        return detect_fake_news(news_text)
    except Exception as e:
        return f"Error: Unable to fetch URL. Details: {str(e)}", None, None

# Fake News Detection
def detect_fake_news(news_text):
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = fake_news_model(**inputs)
    prob = torch.sigmoid(outputs.logits).squeeze().numpy()
    result = "Fake News" if prob > 0.5 else "Real News"
    return result, float(np.round(prob, 2)), news_text

# Load Deepfake Model (Placeholder for now)
@st.cache_resource
def load_deepfake_model():
    return "Deepfake Model Placeholder"

deepfake_model = load_deepfake_model()

# Deepfake Video Analysis
def analyze_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count, fake_frames = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:  # Simulating deepfake detection every 10 frames
            if np.random.rand() > 0.5:
                fake_frames += 1

    cap.release()
    fake_percentage = (fake_frames / frame_count) * 100 if frame_count > 0 else 0
    result = "Fake Video" if fake_percentage > 50 else "Real Video"
    return result, np.round(fake_percentage, 2)

# UI Layout
st.title("🛡️ AI Detection Suite: Fake News & Deepfake Video Analyzer")

tab1, tab2 = st.tabs(["📜 Fake News Detection", "🎥 Deepfake Video Analysis"])

# Fake News Detection Tab
with tab1:
    st.header("📰 Fake News Detection")
    news_text = st.text_area("Enter News Article", "")
    news_url = st.text_input("Enter News URL")

    if st.button("Fetch and Analyze from URL"):
        if news_url.strip():
            with st.spinner("Fetching and analyzing news..."):
                result, confidence, extracted_text = fetch_and_analyze_news(news_url)
                if extracted_text:
                    st.text_area("Extracted News Content", extracted_text, height=200)
                    st.success(f"📰 Analysis Result: **{result}** (Confidence: {confidence * 100}%)")
                else:
                    st.warning(result)
        else:
            st.warning("Please enter a valid URL.")

    if st.button("Analyze News Text"):
        if news_text.strip():
            with st.spinner("Analyzing news text..."):
                result, confidence, _ = detect_fake_news(news_text)
                st.success(f"📰 Analysis Result: **{result}** (Confidence: {confidence * 100}%)")
        else:
            st.warning("Please enter news text for analysis.")

# Deepfake Video Analysis Tab
with tab2:
    st.header("🎭 Deepfake Video Analyzer")
    uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    video_path = None

    if uploaded_video:
        # Temporary save uploaded video to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_video.read())
            video_path = tmpfile.name

        # Use OpenCV to read first frame as a thumbnail
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.image(frame, caption="Uploaded Video Frame", use_column_width=True)

    if video_path and st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            result, fake_percentage = analyze_deepfake(video_path)
            st.success(f"🎭 Analysis Result: **{result}** (Fake Frames: {fake_percentage}%)")

# Footer
st.markdown("""
---
🔍 **Features**: Real-time news verification, deepfake detection, and automatic analysis from URLs.
📌 **Tip**: Use credible sources for better accuracy.
""")
