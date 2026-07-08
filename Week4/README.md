# Week 4: The Web Dashboard

**Focus:** Streamlit and software architecture.

This week the notebook pipeline became a web application. The logic was refactored into a `MemeAnalyzer` class in a separate backend file with no print statements (Assignment 4.1), and a Streamlit frontend was built on top of it with an image uploader and a side-by-side results layout using `st.columns` (Assignment 4.2). `@st.cache_resource` keeps the EasyOCR model loaded across reruns so the app does not freeze on every interaction.

## Files

- `meme_engine.py` — backend `MemeAnalyzer` class (OCR + sentiment logic)
- `app.py` — Streamlit frontend
- `Screenshot 2026-01-07 *.png` — screenshots of the app running locally
