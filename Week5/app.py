import streamlit as st
from meme_engine import MemeAnalyzer
from PIL import Image
import numpy as np

st.set_page_config(page_title="Meme Sentiment Dashboard", layout="wide")

# Cache OCR model
@st.cache_resource
def load_analyzer():
    return MemeAnalyzer()

analyzer = load_analyzer()

st.title("Meme Sentiment Analysis Dashboard")
st.write("Extracting insight from memes using Machine Vision and NLP.")

uploaded_file = st.file_uploader("Upload a Meme", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Meme", use_container_width=True)

    with col2:
        st.subheader("Pipeline Output")

        with st.spinner("Running OCR + Sentiment Analysis..."):
            img_array = np.array(image)
            data = analyzer.analyze_image(img_array)

        st.write(f"**Extracted Text:** {data['raw_text']}")
        st.write(f"**Cleaned Text:** {data['cleaned_text']}")
        st.write(f"**OCR Confidence:** {data['confidence'] * 100:.1f} %")

        m1, m2 = st.columns(2)
        m1.metric("Polarity", data['polarity'])
        m2.metric("Subjectivity", data['subjectivity'])

        if data['polarity'] > 0.1:
            st.success("Result: Positive Sentiment 😊")
        elif data['polarity'] < -0.1:
            st.error("Result: Negative Sentiment 😔")
        else:
            st.info("Result: Neutral Sentiment 😐")