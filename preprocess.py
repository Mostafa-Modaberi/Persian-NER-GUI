from settings import *
from parsivar import Normalizer, Tokenizer, SpellCheck
from tkinter import messagebox

class PersianTextProcessor:
    def __init__(self, spell_check_var):

        self.spell_check_var = spell_check_var
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        self.spell_checker = SpellCheck()
        self.persian_punctuation = ['،', '؛', '؟', '!', '.', ':', '(', ')', '?']

    def separate_punctuation(self, text):
        """Separate Persian punctuation marks from text"""
        for punc in self.persian_punctuation:
            text = text.replace(punc, f' {punc} ')
        return text

    def process_text(self, text):


        # Remove newlines and replace with space
        lines = text.split('\n')
        processed_lines = []
        
        # Process each line to ensure it ends with a period
        for line in lines:
            line = line.strip()
            if line:  # Only process non-empty lines
                if not line.endswith(('.', '،', '؛', '!', '؟', '?')):
                    line = line + ' .'
                processed_lines.append(line)
        
        # Join lines back together with spaces
        text = ' '.join(processed_lines)

        # Replace '؟' with '?'
        text = text.replace('؟', '?')
        # Replace '؛' with '.'
        text = text.replace('؛', '.')
        
        if self.spell_check_var:
            spell_checked_text = self.spell_checker.spell_corrector(text)
        else:
            spell_checked_text = text


        separated_text = self.separate_punctuation(spell_checked_text)
        separated_text_added_space = separated_text + ' '
        
        # Normalize text
        normalized_text = self.normalizer.normalize(separated_text_added_space)

        # handling exception
        normalized_text = normalized_text.replace('بود', ' بود')
        
        # Tokenize sentences
        sentences = self.tokenizer.tokenize_sentences(normalized_text)
        
        # Add period if sentence doesn't end with proper punctuation
        final_sentences = []
        for sentence in sentences:
            if not sentence.strip().endswith(('.', '،', '؛', '!', '؟', '?')):
                sentence = sentence.strip() + ' . '
            final_sentences.append(sentence)
        
        # Tokenize words
        words = self.tokenizer.tokenize_words(normalized_text)
        
        return {
            'separated_text': separated_text,
            'normalized_text': normalized_text,
            'sentences': final_sentences,  # Use the modified sentences
            'words': words
        }