import numpy as np
import tensorflow as tf
import os
import pickle
from use_model import NERPredictor

class NERModel:
    def __init__(self, model, training_data_path, weights_path, text):
        self.model = model
        self.training_data_path = training_data_path
        self.weights_path = weights_path
        # Add any model initialization here
        predictor = NERPredictor(model,training_data_path, weights_path)
        # Check if text is a list and handle accordingly
        # if isinstance(text, list):
        #     self.results = [[(word, int(label)) for word, label in predictor.predict(t)] for t in text]
        # else:
        #     self.results = [(word, int(label)) for word, label in predictor.predict(text)]

        if isinstance(text, list):
            self.results = [[(word, 0) if len(word) <= 2 else (word, int(label)) for word, label in predictor.predict(t)] for t in text]
        else:
            self.results = [(word, 0) if len(word) <= 2 else (word, int(label)) for word, label in predictor.predict(text)]
             
    def predict(self, text: str) -> dict:
        return self.results