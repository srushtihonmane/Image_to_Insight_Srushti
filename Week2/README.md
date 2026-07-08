# Week 2: Machine Vision (OCR)

**Focus:** Optical Character Recognition with EasyOCR.

This week's task was extracting raw text from meme images using EasyOCR's deep learning pipeline. The notebook runs OCR on three different memes (Assignment 2.1) and then filters the results by confidence score, keeping only detections above 0.5 to drop low-quality "garbage" reads (Assignment 2.2).

## Files

- `Assignment2.ipynb` — OCR extraction and confidence filtering
- `meme1.jpg`, `meme2.jpg`, `meme3.jpg` — test meme images
- `output.txt` — extracted text output from the three memes
