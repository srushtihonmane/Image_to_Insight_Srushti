"""
app.py
------
Streamlit frontend for the Image to Insight Meme Analyzer.

Run with:
    streamlit run app.py
"""

import time

import streamlit as st
from PIL import Image

from meme_engine import MemeAnalyzer

st.set_page_config(
    page_title="Meme Analyzer | Image to Insight",
    page_icon="🖼️",
    layout="wide",
)


# The analyzer holds the EasyOCR model (large and slow to load), so it is
# cached once per session instead of being rebuilt on every rerun.
@st.cache_resource
def load_analyzer():
    return MemeAnalyzer(use_gpu=True)


try:
    with st.spinner("Loading AI models (EasyOCR + TextBlob)..."):
        analyzer = load_analyzer()
except Exception as e:
    st.error(f"Could not load the AI models: {e}")
    st.stop()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("Image to Insight")
    st.write(
        "A multimodal pipeline that reads the text inside a meme "
        "and scores the emotion behind it."
    )
    st.markdown("---")
    st.markdown("### Settings")
    enable_spellcheck = st.checkbox(
        "Enable spelling correction",
        value=False,
        help="Fixes OCR spelling mistakes with TextBlob before scoring. "
        "More accurate on noisy text, but slower.",
    )
    st.markdown("---")
    st.caption("Built with EasyOCR, TextBlob and Streamlit.")

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
st.title("🖼️ Meme Sentiment Analysis Dashboard")
st.write(
    "Upload a meme to extract its text with deep learning OCR "
    "and analyze its sentiment with NLP."
)

with st.expander("ℹ️ How to read the scores"):
    st.markdown(
        """
**Polarity (-1.0 to +1.0)** — the emotional direction of the text.

| Range | Sentiment | Typical meme goal |
|-------|-----------|-------------------|
| 0.1 to 1.0 | Positive 😊 | To make you smile |
| -0.1 to 0.1 | Neutral 😐 | To state facts |
| -1.0 to -0.1 | Negative 😔 | To vent or complain |

**Subjectivity (0.0 to 1.0)** — how opinionated the text is.

| Range | Meaning |
|-------|---------|
| 0.6 to 1.0 | Opinion / humour |
| 0.3 to 0.6 | Mix of fact and opinion |
| 0.0 to 0.3 | Factual / objective |
"""
    )

uploaded_file = st.file_uploader("Upload a meme", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Choose a JPG or PNG meme above to get started.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Meme")
        try:
            image = Image.open(uploaded_file)
            st.image(image, width="stretch")
        except Exception as e:
            st.error(f"Could not read this image: {e}")
            st.stop()

    with col2:
        st.subheader("Analysis")

        if st.button("Analyze Meme", type="primary"):
            start = time.time()
            with st.spinner("Running OCR + sentiment analysis..."):
                data = analyzer.analyze_image(image, run_spell_check=enable_spellcheck)
            elapsed = time.time() - start

            if data["status"] == "success":
                polarity = data["polarity"]
                subjectivity = data["subjectivity"]

                if polarity > 0.1:
                    verdict, delta_color = "Positive 😊", "normal"
                elif polarity < -0.1:
                    verdict, delta_color = "Negative 😔", "inverse"
                else:
                    verdict, delta_color = "Neutral 😐", "off"

                m1, m2, m3 = st.columns(3)
                m1.metric("Processing Time", f"{elapsed:.2f}s")
                m2.metric("OCR Confidence", f"{data['confidence'] * 100:.1f}%")
                m3.metric("Sentiment", verdict, f"{polarity:.2f}", delta_color=delta_color)

                st.markdown("#### Extracted Text")
                st.text_area("Raw OCR output", data["raw_text"], height=90)
                st.write(f"**Cleaned text:** {data['cleaned_text']}")

                # Combine polarity and subjectivity into a rough "why was
                # this meme made" classification.
                if polarity > 0.3 and subjectivity > 0.5:
                    purpose = "To make you smile (wholesome / funny)"
                elif polarity < -0.3 and subjectivity > 0.5:
                    purpose = "To vent or complain (relatable frustration)"
                elif polarity > 0 and subjectivity < 0.3:
                    purpose = "To motivate or inform positively"
                elif polarity < 0 and subjectivity < 0.3:
                    purpose = "To warn or critique (serious message)"
                elif abs(polarity) <= 0.1 and subjectivity > 0.5:
                    purpose = "To make you think"
                elif abs(polarity) <= 0.1 and subjectivity < 0.3:
                    purpose = "To state facts (informational)"
                else:
                    purpose = "To entertain (general humour)"

                st.markdown("#### Insight")
                if subjectivity > 0.5:
                    st.info(
                        f"**Subjectivity {subjectivity:.2f}** — opinionated: the text "
                        "expresses a personal feeling or joke rather than plain fact."
                    )
                else:
                    st.info(
                        f"**Subjectivity {subjectivity:.2f}** — objective: the text "
                        "mostly presents information as fact."
                    )
                st.success(f"🎯 **Meme purpose:** {purpose}")

            elif data["status"] == "no_text_found":
                st.warning(
                    "No readable text was found in this image. It may be purely "
                    "visual, or the text is too stylized for the OCR model."
                )

            else:
                st.error(f"Analysis failed: {data.get('message')}")
