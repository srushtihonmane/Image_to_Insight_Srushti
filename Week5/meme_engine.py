import easyocr
from textblob import TextBlob
import re

class MemeAnalyzer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_data):
        results = self.reader.readtext(image_data)

        texts = []
        total_conf = 0
        count = 0

        for (bbox, text, prob) in results:
            if prob > 0.5:
                texts.append(text)
                total_conf += prob
                count += 1

        full_text = " ".join(texts)
        avg_confidence = total_conf / count if count > 0 else 0

        return full_text, avg_confidence

    def analyze_sentiment(self, text):
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        if not clean_text.strip():
            return {
                "raw_text": text,
                "cleaned_text": "No valid text found",
                "polarity": 0.0,
                "subjectivity": 0.0
            }

        blob = TextBlob(clean_text)

        return {
            "raw_text": text,
            "cleaned_text": clean_text,
            "polarity": round(blob.sentiment.polarity, 2),
            "subjectivity": round(blob.sentiment.subjectivity, 2)
        }

    def analyze_image(self, image_data):
        text, confidence = self.extract_text(image_data)

        sentiment_data = self.analyze_sentiment(text)

        sentiment_data["confidence"] = round(confidence, 2)

        return sentiment_data