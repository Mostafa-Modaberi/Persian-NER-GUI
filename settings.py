import customtkinter as ctk
import os

# size 
APP_SIZE = (1100, 650)
APP_MIN_SIZE = (1050, 600)
APP_MAX_SIZE = (1200, 700)
APP_TITLE = 'Persian NER'
ICON_PATH = os.path.join(os.getcwd(), "icon\\icon.ico")

# text
FONT_NORMAL_PERSIAN = 'Shabnam FD'
OUTPUT_FONT_SIZE = 17
NORMAL_FONT_SIZE = 13
SMALL_FONT_SIZE = 11
TEXTBOX_FONT_SIZE = 15
TEXTBOX_FONT_SIZE_OUTPUT = 19
FONT_ENGLISH = 'Ubuntu'

# default values
SPELL_CHECK_DEFAULT = False

MODEL_PATH = {
    'BiLSTM': {
        'weights': os.path.join(os.getcwd(), "weights\\version_2_0_0.weights.h5"),
        'training_data': os.path.join(os.getcwd(), "trainingData\\version_2_0_0_training_data.pkl"),
        'model_explanation': 'A model that uses 2 Layers of bidirectional LSTM.',
        'model_validation_score': '83.85%\n67.56%\n72.45%\n63.30%',
        'model_validation_text': 'تشخیص کامل جملات\nاف1-اسکور\nپرسیژن\nریکال'
    },
    'BiLSTM-CNN-Focalloss-Embedding': {
        'weights': os.path.join(os.getcwd(), "weights\\ner_model_with_preprocessor_pretrained_embedding_cnn_focalloss_weights..weights.h5"),
        'training_data': os.path.join(os.getcwd(), "trainingData\\ner_model_with_preprocessor_pretrained_embedding_cnn_focalloss_training_data.pkl"),
        'model_explanation': 'A model that uses 2 Layers of bidirectional LSTM with CNN and Focal Loss.',
        'model_validation_score': '86.04%\n69.77%\n72.71%\n67.07%',
        'model_validation_text': 'تشخیص کامل جملات\nاف1-اسکور\nپرسیژن\nریکال'
    },
    'BiLSTM-CNN-Transformer': {
        'weights': os.path.join(os.getcwd(), "weights\\version_5_0_3.weights.h5"),
        'training_data': os.path.join(os.getcwd(), "trainingData\\version_5_0_3_training_data.pkl"),
        'model_explanation': 'A model that uses 2 Layers of bidirectional LSTM with CNN and Transformer.',
        'model_validation_score': '83.25%\n67.80%\n72.42%\n63.74%',
        'model_validation_text': 'تشخیص کامل جملات\nاف1-اسکور\nپرسیژن\nریکال'
    },
    'Aggregate of Models': {
        'weights': None,
        'training_data': None,
        'model_explanation': 'Combination of all models.',
        'model_validation_score': 'Combination of all models.',
        'model_validation_text': 'مجموعه ای از مدل ها'
    },
}

# git
GITHUB_URL = 'https://github.com/Mostafa-Modaberi/Persian-NER-GUI'

# images
IMAGE_PATHES = {
    'explanation': os.path.join(os.getcwd(), "images\\explanation.png"),
    'comparison': os.path.join(os.getcwd(), "images\\comparison.png")
}

LIMIATION = '* در این نسخه، جملات طولانی تر از 20 کلمه پردازش نمیشوند'

# colors
BACKGROUND = '#242424'
WHITE = '#FFFFFF'
BLACK = '#000000'
LIGHT_GREY = '#bdc3c7'
GREY = 'grey'
BLUE = '#1f6aa5'
LIGHT_BLUE = '#3399cc'
DARK_GREY = '#4a4a4a'
CLOSE_RED = '#8a0606'
SLIDER_BG = '#64686b'
CLEAR_BUTTON_BG = '#4a4a4a'
DROPDOWM_MAIN_COLOR = '#444'
DROPDOWM_HOVER_COLOR = '#333'


