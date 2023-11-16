import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
import streamlit as st
import tensorflow as tf
from data_preprocessing import DataPreprocessor
from train_test import TextClassifier
import pandas as pd

# Load the saved model
loaded_model_path = "single_lstm_model"
loaded_model = tf.keras.models.load_model(loaded_model_path)

# Set a threshold for classification
threshold = 0.6

# Load data and initialize TextClassifier and DataPreprocessor
df = pd.read_csv("data.csv")
training_size = 25000
vocab_size = 30000
max_length = 15
embedding_dim = 16
lstm_dim = 64
classifier = TextClassifier(embedding_dim, lstm_dim, vocab_size, max_length)
dp = DataPreprocessor(vocab_size, max_length)

# Split the sentences and labels
training_sentences = df['tweet'][0:training_size]
testing_sentences = df['tweet'][training_size:]
training_labels = df['class'][0:training_size]
testing_labels = df['class'][training_size:]

training_padded, testing_padded, training_labels, testing_labels = dp.preprocess_data(
    training_sentences, testing_sentences, training_labels, testing_labels
)

# Streamlit App
st.title("Text Classification App")

# User input for new text
new_text = st.text_input("Enter a new text:")

if st.button("Make Prediction"):
    # Clean and preprocess the new text
    cleaned_text = dp.clean_text(new_text)
    padded_text = dp.cleanup_text(cleaned_text)

    # Make predictions
    predictions = loaded_model.predict(padded_text)

    # Interpret the prediction based on the threshold
    offensive_probability = predictions[0][0]
    if offensive_probability > threshold:
        st.write("The text is predicted to be offensive.")
        st.write("Offensive Probability:", offensive_probability)
    else:
        st.write("The text is predicted to be inoffensive.")
        st.write("Offensive Probability:", offensive_probability)
