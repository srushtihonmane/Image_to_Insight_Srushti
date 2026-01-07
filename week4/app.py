import streamlit as st
from meme_engine import MemeAnalyzer
from PIL import Image
import numpy as np

# Week 4: Software Architecture - Separation of UI and Logic
st.set_page_config(page_title="Meme Sentiment Dashboard", layout="wide")

# Caching: Load the EasyOCR model only once
@st.cache_resource
def load_analyzer():
    return MemeAnalyzer()

analyzer = load_analyzer()

st.title("Meme Sentiment Analysis Dashboard")
st.write("Extracting insight from memes using Machine Vision and NLP.")

# Step 1: Input (Load Image)
uploaded_file = st.file_uploader("Upload a Meme", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Pillow: Resize/Load
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Meme", use_container_width=True)

    with col2:
        st.subheader("Pipeline Output")
        with st.spinner("Processing through Linear Pipeline..."):
            # Step 2: Extract Text (EasyOCR)
            img_array = np.array(image)
            raw_extracted = analyzer.extract_text(img_array)
            
            # Step 3: Analyze Sentiment (TextBlob)
            data = analyzer.analyze_sentiment(raw_extracted)

        # Output: Display extracted text and scores
        st.write(f"**Extracted Text:** {data['raw_text']}")
        st.write(f"**Cleaned Text:** {data['cleaned_text']}")
        
        # Metrics: Polarity and Subjectivity
        m1, m2 = st.columns(2)
        m1.metric("Polarity", data['polarity'], help="Negative (-1) to Positive (+1)")
        m2.metric("Subjectivity", data['subjectivity'], help="Fact (0) to Opinion (1)")

        # Final Vibe Classification
        if data['polarity'] > 0.1:
            st.success("Result: Positive Sentiment")
        elif data['polarity'] < -0.1:
            st.error("Result: Negative Sentiment")
        else:
            st.info("Result: Neutral Sentiment")