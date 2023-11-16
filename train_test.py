import tensorflow as tf
import numpy as np

class TextClassifier:
    def __init__(self, embedding_dim, lstm_dim, vocab_size,max_length):
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Initialize the model
        self.model = self.build_model()
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_dim)),
            #tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        return model
            
    
    def train(self, training_padded, training_labels, testing_padded, testing_labels, num_epochs=10):
        history = self.model.fit(
            training_padded, training_labels,
            epochs=num_epochs,
            validation_data=(testing_padded, testing_labels),
            verbose=2
        )
        return history
    

    
    def save(self, path):
        print("=======saving model=========")
        self.model.save(path)
        print("=======model has been saved successfully==========")

        
    def load(self, path):
        # Load a saved model from a file
        self.loaded_model = tf.keras.models.load_model(path)
        print("=====model has been loaded successfully=======")
        