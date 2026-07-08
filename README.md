# Image to Insight: The Meme Analyzer 🖼️ ➡️ 📊

**Final Project Submission | Image to Insight (NLP & Computer Vision Track)**

## Project Overview

The Meme Analyzer is a multimodal AI web application that reads the text hidden inside meme images and determines the emotional tone behind it. It combines **Computer Vision** and **Natural Language Processing** into a single pipeline: an image goes in, and a sentiment report comes out.

The project was developed over five weeks, evolving from basic image preprocessing into a complete AI-powered web application built with:

- **Pillow** — image loading and preprocessing
- **EasyOCR** — deep learning based Optical Character Recognition
- **TextBlob** — lexicon-based sentiment analysis
- **Streamlit** — interactive web dashboard

## Features

- **Deep learning OCR** that detects and reads text on complex meme backgrounds
- **Confidence filtering** that drops low-quality detections (below 50% probability) so noise never reaches the sentiment stage
- **Sentiment metrics** — polarity (positive/negative) and subjectivity (fact/opinion) for every meme
- **Meme purpose classification** that combines both scores into a human-readable verdict ("to make you smile", "to vent", "to state facts", ...)
- **GPU auto-detection** — uses CUDA when available and falls back to the CPU automatically
- **Optional spelling correction** to clean up noisy OCR output before scoring
- **Model caching** so the heavy EasyOCR model loads only once per session
- **Graceful edge-case handling** for images with no readable text and for unexpected errors

## How the Pipeline Works

```
Meme image ──▶ Preprocessing ──▶ OCR ──▶ Confidence filter ──▶ Text cleaning ──▶ Sentiment analysis ──▶ Dashboard
               (Pillow/NumPy)   (EasyOCR)     (> 0.5)            (regex)           (TextBlob)          (Streamlit)
```

1. **Input** — a meme image is uploaded through the web interface.
2. **Preprocessing** — the image is converted to an RGB array the OCR model can consume.
3. **Text recognition** — EasyOCR detects text regions and recognizes the characters, returning a confidence score for every block.
4. **Filtering & cleaning** — detections below the confidence threshold are dropped; the remaining text is lowercased and stripped of non-letter characters.
5. **Sentiment analysis** — TextBlob scores the cleaned text for polarity and subjectivity.
6. **Dashboard** — the results are displayed with metrics, the extracted text, and an interpretation of what the meme is trying to do.

## Project Structure

```
Image_to_Insight_Srushti/
├── app.py                 # Final Streamlit frontend
├── meme_engine.py         # Final backend engine (MemeAnalyzer class)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── MIDTERM_README.md      # Mid-term project summary
├── ENDTERM_README.md      # End-term project summary
├── Week1/                 # Image preprocessing with Pillow
├── Week2/                 # OCR extraction with EasyOCR
├── Week3/                 # Sentiment analysis + mid-sem prototype
├── Week4/                 # First Streamlit dashboard
└── Week5/                 # End-term submission snapshot
```

## The Journey, Week by Week

| Week | Focus | Deliverable |
|------|-------|-------------|
| [Week 1](Week1/) | Images as matrices; grayscale + resize preprocessing | `Assignment1.ipynb`, `processed_week1.jpg` |
| [Week 2](Week2/) | OCR with EasyOCR and confidence filtering | `Assignment2.ipynb`, `output.txt` |
| [Week 3](Week3/) | TextBlob sentiment; full linear pipeline | `mid_term_prototype.ipynb` |
| [Week 4](Week4/) | Streamlit dashboard; backend/frontend separation | `app.py`, `meme_engine.py`, screenshots |
| [Week 5](Week5/) | End-term submission | Final app snapshot |

The polished final application at the repository root builds on the Week 5 submission with GPU auto-detection, structured error handling, optional spelling correction, and a richer interface.

## Installation & Setup

### Prerequisites

- Python 3.9+
- (Optional) An NVIDIA GPU with CUDA drivers for faster OCR

### Quickstart (Windows)

1. **Clone the repository and open a terminal in it:**
   ```powershell
   git clone https://github.com/srushtihonmane/Image_to_Insight_Srushti.git
   cd Image_to_Insight_Srushti
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
3. **Install the dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Download the TextBlob corpora (one time):**
   ```powershell
   python -m textblob.download_corpora
   ```
5. **Run the app:**
   ```powershell
   streamlit run app.py
   ```
   The dashboard opens at http://localhost:8501.

> **Note:** on the very first analysis EasyOCR downloads its detection and recognition models (about 100 MB). Later runs use the cached models.

## Usage Guide

1. **Upload** a meme (JPG/PNG) with the file uploader.
2. **Optionally** enable spelling correction in the sidebar if the meme text is noisy.
3. **Click "Analyze Meme"** — the first run takes a few seconds while the models warm up.
4. **Read the results** — processing time, OCR confidence, sentiment verdict, the extracted text, and the meme-purpose insight.

## Understanding the Scores

### Polarity (-1.0 to +1.0)

| Range | Sentiment | Typical meme goal |
|-------|-----------|-------------------|
| 0.1 to 1.0 | Positive 😊 | To make you smile (wholesome/funny) |
| -0.1 to 0.1 | Neutral 😐 | To state facts (informational) |
| -1.0 to -0.1 | Negative 😔 | To vent or complain (relatable frustration) |

### Subjectivity (0.0 to 1.0)

| Range | Meaning |
|-------|---------|
| 0.6 to 1.0 | Opinion or humour — expresses personal feelings |
| 0.3 to 0.6 | A mix of fact and opinion |
| 0.0 to 0.3 | Factual and objective |

### The Sarcasm Gap

A known limitation of lexicon-based analysis: a meme can pair positive words ("Great job!") with an image that flips the meaning. TextBlob only sees the text, so heavy sarcasm can be misread. Handling this would require a multimodal model that looks at the image and the text together — a natural next step for this project.

## Technical Architecture

### Backend — `meme_engine.py`

The `MemeAnalyzer` class encapsulates the whole analysis pipeline:

- Loads the EasyOCR reader once (a heavy resource) with automatic GPU/CPU selection
- Accepts file paths, PIL images, or NumPy arrays as input
- Filters OCR detections by confidence and averages the scores of the kept lines
- Cleans the text and scores it with TextBlob, with optional spelling correction
- Returns a structured result dictionary with a `status` field (`success`, `no_text_found`, or `error`) instead of printing or crashing

### Frontend — `app.py`

- `@st.cache_resource` keeps a single `MemeAnalyzer` instance alive across Streamlit reruns
- Side-by-side layout: the uploaded meme next to its analysis
- Metrics row (processing time, OCR confidence, sentiment), extracted text, and a color-coded insight panel
- Distinct handling for the "no text found" and error cases

### Design Choices

- **Separation of concerns** — the engine has no UI code and the UI has no analysis logic, so the backend can be reused from a notebook or script unchanged
- **Single model instance** — the OCR model is loaded once and shared, keeping interactions fast
- **Graceful degradation** — no GPU, no text, or a corrupted image all produce a clear message instead of a crash

## Troubleshooting

- **GPU not being used?** Check `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, install a CUDA build of PyTorch.
- **Slow analysis?** 2–10 seconds per image on CPU is normal; spelling correction adds a few more seconds. Disable it in the sidebar for speed.
- **Import errors?** Re-run `pip install --upgrade -r requirements.txt` inside the virtual environment.
- **`MissingCorpusError` from TextBlob?** Run `python -m textblob.download_corpora`.

## Credits

- **EasyOCR** by JaidedAI — text detection and recognition models
- **TextBlob** by Steven Loria — sentiment analysis
- **Streamlit** — the web application framework
- **Image to Insight** mentors Santosh and Lakshaditya — curriculum, weekly resources, and reviews ([central hub repository](https://github.com/santoshguntuku/Image-to-Insight))

---

Developed by **Srushti Honmane** as the final submission for the Image to Insight project.
