import tensorflow as tf
from data_preprocessing import DataPreprocessor
from train_test import TextClassifier
import pandas as pd


training_size = 25000
vocab_size = 30000
max_length = 15
embedding_dim = 16
lstm_dim = 64

df = pd.read_csv("data.csv")

# Load the saved model
loaded_model_path = "C:/Program Files/GitHub/My repositories/restructured/single_lstm_model"
loaded_model = tf.keras.models.load_model(loaded_model_path)

# Set a threshold for classification
threshold = 0.6

classifier = TextClassifier(embedding_dim, lstm_dim, vocab_size, max_length)
dp = DataPreprocessor(vocab_size, max_length)

training_sentences = df['tweet'][0:training_size]
testing_sentences = df['tweet'][training_size:]





# Split the labels
training_labels = df['class'][0:training_size]
testing_labels =  df['class'][training_size:]

training_padded, testing_padded, training_labels, testing_labels = dp.preprocess_data(
        training_sentences, testing_sentences, training_labels, testing_labels
    )



# Make predictions
new_text = "hate is a crime"
padded_new_text = dp.cleanup_text(new_text)  # Assuming this method is defined within the TextClassifier class
predictions = loaded_model.predict(padded_new_text)




# Interpret the prediction based on the threshold
offensive_probability = predictions[0][0]
if offensive_probability > threshold:
    print("The text is predicted to be offensive.")
    print(offensive_probability)
else:
    print("The text is predicted to be inoffensive.")
    print(offensive_probability)

