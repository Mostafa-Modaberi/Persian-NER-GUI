import pickle
import tensorflow as tf
from ner_model import NERModelCNNFocallossEmbedding, NERModelBase, NERModelTransformer
import arabic_reshaper
from bidi.algorithm import get_display
from settings import *

class NERPredictor: 
    def __init__(self, model, training_data_path, weights_path):
        """Initialize the NER predictor with paths to required files"""
        self.model = model
        self.training_data_path = training_data_path
        self.weights_path = weights_path
        self._initialize_model()
    
    def _setup_model(self, model, training_data, weights_path):
        """Initialize, compile and load weights for a model"""
        model.initialize(training_data)
        model.compile_model()
        model.load_weights(weights_path)
        return model

    def _initialize_model(self):
        """Initialize and compile the model with pre-trained weights"""

        # Create and initialize model
        match self.model:
            case 'BiLSTM-CNN-Focalloss-Embedding':
                # Load training data and configurations
                with open(self.training_data_path, 'rb') as f:
                    training_data = pickle.load(f)
                self.ner_model = self._setup_model(NERModelCNNFocallossEmbedding(), training_data, self.weights_path)
            case 'BiLSTM':
                # Load training data and configurations
                with open(self.training_data_path, 'rb') as f:
                    training_data = pickle.load(f)
                self.ner_model = self._setup_model(NERModelBase(), training_data, self.weights_path)
            case 'BiLSTM-CNN-Transformer':
                # Load training data and configurations
                with open(self.training_data_path, 'rb') as f:
                    training_data = pickle.load(f)
                self.ner_model = self._setup_model(NERModelTransformer(), training_data, self.weights_path)
            case 'Aggregate of Models':
                # Initialize all three models

                base_weights_path = MODEL_PATH['BiLSTM']['weights']
                cnn_weights_path = MODEL_PATH['BiLSTM-CNN-Focalloss-Embedding']['weights']
                transformer_weights_path = MODEL_PATH['BiLSTM-CNN-Transformer']['weights']

                base_training_data_path = MODEL_PATH['BiLSTM']['training_data']
                cnn_training_data_path = MODEL_PATH['BiLSTM-CNN-Focalloss-Embedding']['training_data']
                transformer_training_data_path = MODEL_PATH['BiLSTM-CNN-Transformer']['training_data']

                # Load training data and configurations
                with open(base_training_data_path, 'rb') as f:
                    base_training_data = pickle.load(f)
                with open(cnn_training_data_path, 'rb') as f:
                    cnn_training_data = pickle.load(f)
                with open(transformer_training_data_path, 'rb') as f:
                    transformer_training_data = pickle.load(f)

                self.model_cnn = self._setup_model(NERModelCNNFocallossEmbedding(), cnn_training_data, cnn_weights_path)
                self.model_base = self._setup_model(NERModelBase(), base_training_data, base_weights_path)
                self.model_transformer = self._setup_model(NERModelTransformer(), transformer_training_data, transformer_weights_path)
                self.is_aggregate = True
                return
                
        self.is_aggregate = False
   
    def _predict_ner_tags(self, text, predictions):
        """Convert predictions to list of (word, label) tuples"""
        words = text.split()
        pred_labels = predictions[0].argmax(axis=-1)[:len(words)]
        return list(zip(words, pred_labels))
    
    def predict(self, text):
        """Predict NER tags for given text and return list of (token, label) tuples"""
        if not self.is_aggregate:
            predictions = self.ner_model.predict([text])
            return self._predict_ner_tags(text, predictions)
        
        # Get predictions from all models
        predictions = []
        predictions.append(self.model_cnn.predict([text]))
        predictions.append(self.model_base.predict([text]))
        predictions.append(self.model_transformer.predict([text]))
        
        # Convert predictions to label indices
        words = text.split()
        all_labels = []
        for pred in predictions:
            pred_labels = pred[0].argmax(axis=-1)[:len(words)]
            all_labels.append(pred_labels)
        
        # Majority voting for each word
        final_labels = []
        for i in range(len(words)):
            votes = [labels[i] for labels in all_labels]
            # Get most common label (majority vote)
            final_label = max(set(votes), key=votes.count)
            final_labels.append(final_label)
            
        return list(zip(words, final_labels))
