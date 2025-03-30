

import streamlit as stl
import pandas as pd
import pickle  # To download and load the model that have been trained 
from sklearn.feature_extraction.text import TfidfVectorizer 

stl.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Corrected Paths Using Double Backslashes
model_path = 'C:\\Users\\akshi\\downloads\\project\\fake_news_model.pkl'  # Path of Downloaded model on Kaggle 
vectorizer_path = 'C:\\Users\\akshi\\downloads\\project\\tfidf_vectorizer.pkl'



# Load the model and vectorizer after fixing paths
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)



# Apply CSS commands to make it more interactive, Streamlit does not support much interactive commands 


dark_theme_css = """
<style>
/* Global App Styles */
# It helps to change the color of background 
.stApp {   
    background-color: #1E1E1E !important;
    color: #FFFFFF !important; 
}

# It will work only on the heading part 
h1 {
    font-size: 36px;
    font-weight: bold;
    color: green !important;
    text-align: center;
    margin-bottom: 20px;
}

# It will change the theme and font size of the input we are giving 
.stTextInput>div>div>input {
    background-color: #000;
    color: #fff;
    border-radius: 8px;
    border: 1px solid #555 !important;
    padding: 10px;
    font-size: 16px;
}

/* Button Styles */
.stButton>button {
    background-color: #FF4B4B !important;
    color: white !important;
    padding: 12px 20px;
    border: 2px solid white !important;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
}
.stButton>button:hover {
    background: linear-gradient(to right, #2575fc, #6a11cb) !important;
}

/* Success & Warning Styles */
div[class*="stAlert"] {
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
    font-size: 16px;
}

/* Success Message */
div[class*="stAlert success"] {
    background-color: #0f5132 !important;
    border: 2px solid #28a745 !important;
    color: #d4edda !important;
}

/* Warning Message */
div[class*="stAlert warning"] {
    background-color: #5a3d02 !important;
    border: 2px solid #ffc107 !important;
    color: #fff3cd !important;
}
</style>
"""


stl.markdown(dark_theme_css, unsafe_allow_html=True) # By this Command we can allow CSS styling on our Streamlit Interface 


stl.title("📰 Fake News Detector")


headline = stl.text_input("Enter the News Headline (Title)")


content = stl.text_area("Enter the News Content (Text)")


if stl.button("Detect Fake News"):
    if headline and content:
        # Combine title and content to match training data format
        combined_input = headline + " " + content

       

        input_data = [combined_input]
        input_vectorized = tfidf_vectorizer.transform(input_data)

        # Make prediction
        prediction = model.predict(input_vectorized)[0]
        if prediction == 1:
            stl.success(" This News is **Real!**")
        else:
            stl.error(" This News is **Fake!**")
    else:
        stl.warning(" Please enter both title and content before submitting.")
