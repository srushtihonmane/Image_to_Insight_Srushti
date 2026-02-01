## Meme Sentiment Analysis Project  

This project was developed to create a functional web application capable of reading text from meme images and determining their emotional tone. Over a four-week development period, the system evolved from basic image processing techniques into a complete AI-powered application integrating deep learning models and a web-based dashboard.

The goal of the project was to demonstrate how artificial intelligence can transform visual content into meaningful emotional insights.

---

## How the System Works  

The application follows a structured linear processing pipeline:

**Input**  
A meme image is uploaded through the web interface.

**Image Preprocessing**  
The system resizes and cleans the image to improve readability for the AI models.

**Text Recognition**  
A deep learning-based OCR engine identifies letters and words within the image.

**Sentiment Analysis**  
The extracted text is analyzed to determine whether the emotion is positive, negative, or neutral.

**Web Dashboard**  
The final results are displayed on a clean and interactive web interface.

---

## Weekly Development Breakdown  

### Week 1: Image Preparation  

This phase focused on understanding how computers interpret images as numerical grids. A preprocessing module was developed to convert images into grayscale and resize them for efficient processing. Grayscale conversion removed unnecessary visual information that could interfere with accurate text recognition.

---

### Week 2: OCR Implementation  

EasyOCR was integrated to enable text detection and recognition using deep learning. Since OCR operates probabilistically, a confidence filtering mechanism was added to remove any detected text with less than 50 percent reliability.

---

### Week 3: Emotion Understanding  

TextBlob was used to analyze the emotional tone of extracted text using two main metrics:

- Polarity to identify positive, negative, or neutral sentiment  
- Subjectivity to determine factual versus opinion-based content  

The concept of the “Sarcasm Gap” was also explored, where positive text may express negative meaning when paired with certain images.

---

### Week 4: Web Application Development  

The complete system was transformed into a web application using Streamlit. The backend processing engine was separated from the frontend interface to follow good software engineering practices. Caching was added to keep AI models loaded in memory, ensuring smooth and fast performance.

---

## Repository Files  

**app.py**  
Streamlit-based web interface for user interaction.

**meme_engine.py**  
Core engine handling OCR and sentiment processing.

**requirements.txt**  
List of Python dependencies.

**README.md**  
Project documentation.

