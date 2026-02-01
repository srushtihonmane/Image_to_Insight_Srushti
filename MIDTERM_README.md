Meme Sentiment Analysis Project
I built this project to create a functional web application that reads text from memes and determines their emotional tone. This was a 4-week journey where I moved from basic image processing to implementing deep learning models and a web-based dashboard.

How My System Works
I designed the application to follow a step-by-step process called a linear pipeline:

Input: I upload an image file of a meme.

Image Prep: My system resizes and cleans the image to make it easier for the computer to read.

Reading: I use an OCR engine to identify letters and words within the pixels.

Sentiment Check: I analyze the words to see if they are positive, negative, or neutral.

Dashboard: I display the final results on a clean web interface.

My Weekly Development Breakdown
Week 1: Preparing the Image
My first lesson was understanding the computer vision of images made out of a grid of numbers. To make the AI process faster and more precise, I have developed a pre-processor that converts images into grayscale images and resizes the images. Grayscaling images takes out the redundant information of the image that could confuse the reader engine.

Week 2: Teaching the Computer to Read I added
 another module called EasyOCR that relies on deep learning to locate text. I learned that it’s still a probability because the computer guesses the letters. To guarantee that I have quality text, I added code to filter out any text that the computer isn’t at least 50% sure of.

Week 3: Understanding the Emotion
After that, I obtained the text, in which I analyzed the tone using a library called TextBlob. I concentrated on two scores:
Polarity: This identifies whether the words from the texts are happy, sad, or angry.
Subjectivity: Assessing whether it is a fact claim or simply an opinion claim. Similarly, I analyzed the concept of the ‘Sarcasm Gap,’ referring to the phenomenon when a text is perceived to be positive but the accompanying picture reveals it to be a joke.

Week 4: Creating Web Application In the final stage, I migrated my code from notebooks to an actual web app using the Streamlit library. I also concentrated on software design skills and ensured I separated my engine code (backend) and the graphics user interface code (frontend) in my system design. I applied the caching method to ensure the AI model is always in memory while utilizing the system, so the system does not stop

Files in My Repository
app.py: The code I wrote for the web interface.
meme_engine.py: The "brain" of my project that handles all the heavy AI calculations.
requirements.txt: A list of the Python tools I used to run this project.
README.md: This document providing my project summary.
