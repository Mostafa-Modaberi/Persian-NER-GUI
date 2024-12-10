import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (MultiHeadAttention, Add, LayerNormalization, 
                                   BatchNormalization, Concatenate, Conv1D)
from tensorflow.keras.optimizers import AdamW

class NERModelCNNFocallossEmbedding:
    def __init__(self):
        self.vectorize_layer = None
        self.model = None
        self.vocab_size = None
        self.embedding_dim = None
        self.max_sequence_length = None
        self.num_classes = None
    
    def initialize(self, config_data):
        """Initialize model with configuration data"""
        self.vocab_size = config_data['vocab_size']
        self.embedding_dim = config_data['embedding_dim']
        self.max_sequence_length = config_data['max_sequence_length']
        self.num_classes = config_data['num_classes']
        
        # Create TextVectorization layer
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=config_data['max_tokens'],
            output_mode='int',
            output_sequence_length=self.max_sequence_length
        )
        
        # Set vocabulary directly
        self.vectorize_layer.set_vocabulary(config_data['vocabulary'])
        
        # Build the model
        self.model = self._build_model()
        
        # Set embedding weights
        embedding_layer = self.model.get_layer('embedding')
        embedding_layer.set_weights([config_data['embedding_matrix']])
    
    def _build_model(self):
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            trainable=True,
            name='embedding'
        )(input_layer)
        
        # Initial dropout
        x = Dropout(0.1)(embedding_layer)
        
        # CNN layers
        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
        
        # Combine CNN outputs
        x = tf.keras.layers.Concatenate()([conv1, conv2])
        x = tf.keras.layers.BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(
            units=128,
            return_sequences=True,
            recurrent_dropout=0.1
        ))(x)
        
        # Output layers
        x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(128, activation='relu'))(x)
        x = Dropout(0.1)(x)
        output = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)
        
        return Model(inputs=input_layer, outputs=output)
    
    def sequence_focal_loss(self, gamma=2., alpha=0.75):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
            
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            
            pos_weight = alpha
            neg_weight = 1 - alpha
            
            weights = y_true_one_hot * pos_weight + (1 - y_true_one_hot) * neg_weight
            
            cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
            focal_weight = weights * tf.math.pow(1 - y_pred, gamma)
            focal_loss = focal_weight * cross_entropy
            
            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
        return loss
    
    def compile_model(self, learning_rate=2e-4):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.0
            ),
            loss=self.sequence_focal_loss(gamma=2.0, alpha=0.75)
        )
    
    def load_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
        self.model.load_weights(weights_path)
    
    def predict(self, text):
        """Predict NER tags for input text"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
            
        # Vectorize the input text
        vectorized_text = self.vectorize_layer(text)
        if len(vectorized_text.shape) == 1:
            vectorized_text = tf.expand_dims(vectorized_text, 0)
            
        # Make prediction
        predictions = self.model.predict(vectorized_text)
        return predictions
    
class NERModelBase:
    def __init__(self):
        self.vectorize_layer = None
        self.model = None
        self.vocab_size = None
        self.embedding_dim = None
        self.max_sequence_length = None
        self.num_classes = None
        self.embedding_matrix = None

    def initialize(self, config_data):
        """Initialize model with configuration data"""
        self.vocab_size = config_data['vocab_size']
        self.embedding_dim = config_data['embedding_dim']
        self.max_sequence_length = config_data['max_sequence_length']
        self.num_classes = config_data['num_classes']
        self.embedding_matrix = config_data['embedding_matrix']
        
        # Create TextVectorization layer
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=config_data['max_tokens'],
            output_mode='int',
            output_sequence_length=self.max_sequence_length
        )
        
        # Set vocabulary directly
        self.vectorize_layer.set_vocabulary(config_data['vocabulary'])
        
        # Build the model
        self.model = self.build_model()

    def build_model(self):
        # Define the input layer
        input_layer = Input(shape=(self.max_sequence_length,))

        # Embedding Layer with GloVe embeddings
        model = Embedding(input_dim=self.vocab_size,
                          output_dim=self.embedding_dim,
                          weights=[self.embedding_matrix],
                          input_length=self.max_sequence_length,
                          trainable=False)(input_layer)

        # Dropout Layer for regularization
        model = Dropout(0.2)(model)

        # First Bidirectional LSTM Layer
        model = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.2))(model)

        # Second Bidirectional LSTM Layer
        model = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.2))(model)

        # Adding a Dense Layer to learn more features
        model = Dropout(0.3)(model)
        model = TimeDistributed(Dense(self.num_classes, activation='softmax'))(model)

        return Model(inputs=input_layer, outputs=model)

    def compile_model(self, learning_rate=1e-3):
        """Compile the model with an optimizer and loss function"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy'
        )

    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
        except ValueError as e:
            print("Error loading weights:", e)

    def predict(self, text):
        """Predict NER tags for input text"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
        
        # Vectorize the input text
        vectorized_text = self.vectorize_layer(text)
        if len(vectorized_text.shape) == 1:
            vectorized_text = tf.expand_dims(vectorized_text, 0)
        
        # Make prediction
        predictions = self.model.predict(vectorized_text)
        return predictions

    def summary(self):
        self.model.summary()

class NERModelTransformer:
    def __init__(self):
        self.vectorize_layer = None
        self.model = None
        self.vocab_size = None
        self.embedding_dim = None
        self.max_sequence_length = None
        self.num_classes = None
        self.head_size = None
        self.num_heads = None
        self.ff_dim = None
        self.units = None

    def initialize(self, config_data):
        """Initialize model with configuration data"""
        # پارامترهای اصلی
        self.vocab_size = config_data['vocab_size']
        self.embedding_dim = config_data['embedding_dim']
        self.max_sequence_length = config_data['max_sequence_length']
        self.num_classes = config_data['num_classes']
        
        # پارامترهای ترنسفورمر
        self.head_size = config_data.get('head_size', 64)
        self.num_heads = config_data.get('num_heads', [4, 8])
        self.ff_dim = config_data.get('ff_dim', 256)
        self.units = config_data.get('units', 128)
        
        # Create TextVectorization layer
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=config_data['max_tokens'],
            output_mode='int',
            output_sequence_length=self.max_sequence_length
        )
        
        # Set vocabulary directly
        self.vectorize_layer.set_vocabulary(config_data['vocabulary'])
        
        # Build the model
        self.model = self._build_model()
        
        # Set embedding weights
        embedding_layer = self.model.get_layer('embedding')
        embedding_layer.set_weights([config_data['embedding_matrix']])
        
    def enhanced_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """Create an enhanced transformer block with CNN integration"""
        attention_output = MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(inputs, inputs)
        attention_output = Add()([attention_output, inputs])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        
        # Parallel CNN paths
        conv1 = Conv1D(filters=ff_dim//2, kernel_size=3, padding='same', activation='relu')(attention_output)
        conv2 = Conv1D(filters=ff_dim//2, kernel_size=5, padding='same', activation='relu')(attention_output)
        
        x = Concatenate()([conv1, conv2])
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        x = Dense(ff_dim, activation='relu')(x)
        x = Dense(inputs.shape[-1])(x)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        return x

    def _build_model(self):
        input_layer = Input(shape=(self.max_sequence_length,))
        
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            trainable=True,
            name='embedding'
        )(input_layer)
        
        x = Dropout(0.1)(embedding_layer)
        
        # Enhanced transformer blocks
        x = self.enhanced_transformer_block(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)
        x = self.enhanced_transformer_block(x, head_size=64, num_heads=8, ff_dim=256, dropout=0.1)
        
        # Project to match LSTM dimensions
        x = Dense(256)(x)
        
        # Bidirectional LSTM with residual connections
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x)
        x = Add()([x, lstm1])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        lstm2 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1))(x)
        x = Add()([x, lstm2])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Output layers
        x = TimeDistributed(Dense(256, activation='relu'))(x)
        x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(128, activation='relu'))(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        output = TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)
        
        return Model(inputs=input_layer, outputs=output)

    def sequence_focal_loss(self, gamma=2., alpha=0.75):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            
            # Dynamic class weighting
            pos_samples = tf.reduce_sum(y_true_one_hot[..., 1])
            total_samples = tf.cast(tf.size(y_true), tf.float32)
            pos_weight = tf.maximum(1.0, (total_samples - pos_samples) / (pos_samples + 1e-7))
            
            weights = y_true_one_hot * pos_weight + (1 - y_true_one_hot)
            focal_weights = weights * tf.pow(1 - y_pred, gamma)
            
            cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
            loss = focal_weights * cross_entropy
            
            return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def compile_model(self, learning_rate=1e-4):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
            
        self.model.compile(
            optimizer=AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-5,
                clipnorm=1.0
            ),
            loss=self.sequence_focal_loss(gamma=2.0, alpha=0.75)
        )
    
    def load_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
        self.model.load_weights(weights_path)
    
    def predict(self, text):
        """Predict NER tags for input text"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize first.")
            
        # Vectorize the input text
        vectorized_text = self.vectorize_layer(text)
        if len(vectorized_text.shape) == 1:
            vectorized_text = tf.expand_dims(vectorized_text, 0)
            
        # Make prediction
        predictions = self.model.predict(vectorized_text)
        return predictions