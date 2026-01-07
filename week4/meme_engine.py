import easyocr
from textblob import TextBlob
import re

class MemeAnalyzer:
    def __init__(self):
        # Initializes the probabilistic pipeline (Detection + Recognition)
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_data):
        # Extracting text with confidence scores
        results = self.reader.readtext(image_data)
        
        # Filtering out low-confidence "garbage" results (< 0.5)
        filtered_text = [text for (bbox, text, prob) in results if prob > 0.5]
        
        return " ".join(filtered_text)
    def analyze_sentiment(self, text):
        # Ensure text is lowercase to help TextBlob match words
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Check if the text is empty after cleaning
        if not clean_text.strip():
            return {"raw_text": text, "cleaned_text": "No valid text found", "polarity": 0.0, "subjectivity": 0.0}

        blob = TextBlob(clean_text)
        
        return {
            "raw_text": text,
            "cleaned_text": clean_text,
            "polarity": round(blob.sentiment.polarity, 2),
            "subjectivity": round(blob.sentiment.subjectivity, 2)
        }
        