# import pandas as pd
from data_preprocessing import DataPreprocessor
from train_test import TextClassifier


import pandas as pd
if __name__ == "__main__":
    # Load data and initialize TextClassifier
    df = pd.read_csv("data.csv")  # Load your data here

    
    # Initialize ModelTrainer
    training_size = 25000
    vocab_size = 30000
    max_length = 15
    embedding_dim = 16
    lstm_dim = 64

    # Split the sentences
    training_sentences = df['tweet'][0:training_size]
    testing_sentences = df['tweet'][training_size:]

    # Split the labels
    training_labels = df['class'][0:training_size]
    testing_labels =  df['class'][training_size:]

    classifier = TextClassifier(embedding_dim, lstm_dim, vocab_size,max_length)
    dp = DataPreprocessor(vocab_size, max_length)

    # Preprocess data
   
    df["tweet"] = df["tweet"].apply(dp.clean_text)

    training_sentences = df['tweet'][0:training_size]
    testing_sentences = df['tweet'][training_size:]
    training_labels = df['class'][0:training_size]
    testing_labels = df['class'][training_size:]

    training_padded, testing_padded, training_labels, testing_labels = dp.preprocess_data(
        training_sentences, testing_sentences, training_labels, testing_labels
    )

    # Train the model
    num_epochs = 10
    history = classifier.train(
        training_padded, training_labels, testing_padded, testing_labels, num_epochs=num_epochs
    )
    
    
    classifier.save("single_lstm_model")


