"""
meme_engine.py
--------------
Backend engine for the Image to Insight Meme Analyzer.

The MemeAnalyzer class wraps the two AI stages of the pipeline:
EasyOCR reads the text out of a meme image (Computer Vision) and
TextBlob scores the sentiment of that text (NLP). The frontend
(app.py) only talks to this class, keeping the interface and the
logic completely separate.
"""

import logging
import re

import easyocr
import numpy as np
import torch
from PIL import Image
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MemeAnalyzer:
    """End-to-end meme analysis: image in, sentiment report out.

    Loading the EasyOCR model is expensive, so one instance should be
    created per session and reused. The Streamlit frontend does this
    with @st.cache_resource.
    """

    def __init__(self, languages=None, use_gpu=True, confidence_threshold=0.5):
        """Load the EasyOCR model into memory.

        Args:
            languages: list of language codes for OCR (default: ['en']).
            use_gpu: try to run OCR on the GPU. Falls back to the CPU
                automatically when CUDA is not available.
            confidence_threshold: OCR detections below this probability
                are treated as noise and dropped.
        """
        if languages is None:
            languages = ["en"]

        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()

        logging.info(
            "Initializing MemeAnalyzer (CUDA available: %s, using GPU: %s)",
            torch.cuda.is_available(),
            self.use_gpu,
        )

        try:
            self.reader = easyocr.Reader(languages, gpu=self.use_gpu)
            logging.info("EasyOCR model loaded.")
        except Exception:
            logging.exception("Failed to load the EasyOCR model.")
            raise

    def _to_array(self, image_input):
        """Accept a file path, PIL image or numpy array and return an RGB numpy array."""
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("Expected a file path, a PIL Image or a numpy array.")

        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def extract_text(self, image_input):
        """Run OCR on an image and keep only the confident detections.

        Returns:
            (text_lines, avg_confidence): the detected text blocks above
            the confidence threshold and their average confidence score.
        """
        img_array = self._to_array(image_input)
        results = self.reader.readtext(img_array)

        texts = []
        total_conf = 0.0

        for (bbox, text, prob) in results:
            if prob > self.confidence_threshold:
                texts.append(text)
                total_conf += prob

        avg_confidence = total_conf / len(texts) if texts else 0.0
        return texts, avg_confidence

    def analyze_sentiment(self, text, run_spell_check=False):
        """Score a piece of text with TextBlob.

        Args:
            text: raw OCR output.
            run_spell_check: fix OCR spelling mistakes with TextBlob's
                corrector before scoring. More accurate on noisy text,
                but noticeably slower.

        Returns:
            dict with cleaned_text, polarity (-1.0 to 1.0) and
            subjectivity (0.0 to 1.0).
        """
        clean_text = re.sub(r"[^a-zA-Z\s]", "", text.lower()).strip()

        if not clean_text:
            return {"cleaned_text": "", "polarity": 0.0, "subjectivity": 0.0}

        blob = TextBlob(clean_text)

        if run_spell_check:
            logging.info("Running spelling correction (this can take a few seconds)...")
            blob = blob.correct()
            clean_text = str(blob)

        return {
            "cleaned_text": clean_text,
            "polarity": round(blob.sentiment.polarity, 2),
            "subjectivity": round(blob.sentiment.subjectivity, 2),
        }

    def analyze_image(self, image_input, run_spell_check=False):
        """Run the full pipeline: image -> OCR -> sentiment.

        Args:
            image_input: file path, PIL image or numpy array.
            run_spell_check: forwarded to analyze_sentiment().

        Returns:
            dict with a 'status' key ('success', 'no_text_found' or
            'error'). On success it also carries raw_text, cleaned_text,
            text_lines, confidence, polarity and subjectivity; on error
            it carries a 'message' instead.
        """
        try:
            text_lines, confidence = self.extract_text(image_input)

            if not text_lines:
                logging.warning("No text found above the confidence threshold.")
                return {
                    "status": "no_text_found",
                    "raw_text": "",
                    "cleaned_text": "",
                    "text_lines": [],
                    "confidence": 0.0,
                    "polarity": 0.0,
                    "subjectivity": 0.0,
                }

            raw_text = " ".join(text_lines)
            sentiment = self.analyze_sentiment(raw_text, run_spell_check=run_spell_check)

            logging.info(
                "Analysis complete: %d lines, confidence %.2f, polarity %.2f",
                len(text_lines),
                confidence,
                sentiment["polarity"],
            )

            return {
                "status": "success",
                "raw_text": raw_text,
                "cleaned_text": sentiment["cleaned_text"],
                "text_lines": text_lines,
                "confidence": round(confidence, 2),
                "polarity": sentiment["polarity"],
                "subjectivity": sentiment["subjectivity"],
            }

        except Exception as e:
            logging.exception("Analysis failed.")
            return {"status": "error", "message": str(e)}
