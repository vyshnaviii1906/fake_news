import torchvision.transforms as transforms
from PIL import Image

# ✅ Disable Hugging Face Cache Warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 📌 Load Fake News Detection Model
MODEL_NAME_FAKE_NEWS = "microsoft/deberta-v3-base"
model_fake_news = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FAKE_NEWS)
tokenizer_fake_news = AutoTokenizer.from_pretrained(MODEL_NAME_FAKE_NEWS)

# 🎥 ✅ Load Xception Model from timm (Fix for previous error)
deepfake_model = timm.create_model('xception', pretrained=True)
deepfake_model.eval()

# 🖼 Define image transformations for deepfake detection
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception expects 299x299 images
    transforms.ToTensor(),
])

# 🎭 Deepfake Landmark Detector
device = "cpu"
face_alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

# 📰 **News Article Credibility Analysis**
def analyze_news(article_text):
    inputs = tokenizer_fake_news(article_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_fake_news(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    credibility_score = probs[0][1].item() * 100  # Higher probability = more credible

    return credibility_score

# 🎥 **Deepfake Video Analysis**
def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    frame_count = 0
    deepfake_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = deepfake_model(image)
            deepfake_score = torch.nn.functional.softmax(prediction, dim=-1)[0][1].item()  # Extract probability

        deepfake_flags.append(deepfake_score)

        # ✅ Process only 1 frame per second to optimize performance
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * 30)

    cap.release()

    if frame_count == 0:
        return "Error: No frames detected in video."

    # 🧠 Calculate Deepfake Probability
    avg_deepfake_score = np.mean(deepfake_flags) * 100  # Convert to percentage

    return avg_deepfake_score

# 🎯 **Streamlit UI**
st.title("🕵️ AI-Powered Fake News & Deepfake Detection")

# 📰 **News Article Analysis**
st.subheader("📰 News Article Analysis")
article_text = st.text_area("Paste the news article here:")

if st.button("Analyze News"):
    if article_text:
        credibility_score = analyze_news(article_text)

        st.subheader("📰 News Article Analysis:")
        st.write(f"**Credibility Score: {credibility_score:.2f}%**")

        if credibility_score < 50:
            st.error("❌ This article is **FAKE NEWS** (Low Credibility).")
        else:
            st.success("✅ This article is **REAL NEWS** (High Credibility).")
    else:
        st.warning("⚠️ Please enter a news article.")

# 🎥 **Deepfake Video Analysis**
st.subheader("🎥 Deepfake Video Analysis")
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

if st.button("Analyze Video"):
    if video_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        video_path = "uploaded_video.mp4"
        deepfake_probability = detect_deepfake(video_path)

        st.subheader("🎥 Deepfake Video Analysis:")
        if isinstance(deepfake_probability, str):  # Error handling
            st.error(deepfake_probability)
        else:
            st.write(f"**Deepfake Probability: {deepfake_probability:.2f}%**")

            if deepfake_probability > 50:
                st.error("❌ This video is **FAKE** (Deepfake Suspected).")
            else:
                st.success("✅ This video is **REAL** (Low Probability of Deepfake).")
    else:
        st.warning("⚠️ Please upload a video file.")