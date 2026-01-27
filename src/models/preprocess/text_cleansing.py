import re
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord

from src.dictionary.exclude_words import exclude_stopwords

negation_phrases = [
    'sangat tidak menyukai', 'tidak menyukai', 'sangat tidak suka',
    'tidak suka', 'kurang suka', 'kurang menyukai', 'ga suka',
    'gak suka', 'ga menyukai', 'gak menyukai', 'gada'
]

positive_phrases = ['suka', 'sangat suka', 'menyukai', 'sangat menyukai']

class TextCleansing:
    def __init__(self, text, norm_dict=None, exclude_stopwords=None, add_stopwords=['nya', 'ya', 'nih']):
        self.text = text
        self.norm_dict = norm_dict if norm_dict else {}
        self.exclude_stopwords = exclude_stopwords if exclude_stopwords else []

        self.tokenizer = Tokenizer()
        try:
            self.stopword = StopWord()
            stopword_list = self.stopword.get_stopword()
            self.stopword_list = list(stopword_list) if stopword_list else []
            self.stopword_list.extend(add_stopwords)
        except Exception as e:
            self.stopword = None
            self.stopword_list = list(add_stopwords) if add_stopwords else []

    def correct_typos(self, text):
        for typo, correction in self.norm_dict.items():
            text = re.sub(
                rf'\b{typo}\b', correction, text, flags=re.IGNORECASE
            )
        return text
    
    def reduce_extra_characters(self, text):
        """
        Contoh:
        "sukaaaaa" -> "suka"
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)
    
    def split_nya(self, text, exception_words=None):
        if exception_words is None:
            exception_words = ["tanya", "punya", "bertanya", "hanya"]

        if text in exception_words:
            return text

        return re.sub(r'(.*?)nya$', r'\1 nya', text)

    def process_split_nya(self, text):
        words = text.split()
        processed_words = [
            self.split_nya(word.strip()) for word in words
        ]
        return ' '.join(processed_words)
    
    def clean_text_topic(self, negation=False): 
        if negation: 
            for phrase in negation_phrases:
                self.text = self.text.replace(phrase, '')
        else:
            for phrase in positive_phrases:
                self.text = self.text.replace(phrase, '')
        text = re.sub(r"[^a-zA-Z\s']", ' ', self.text)
        text = re.sub(r'\bga\b', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def text_preprocessing_topic(self):
        tokens = self.tokenizer.tokenize(self.text)
        processed_tokens = [
            word if (word not in self.exclude_stopwords) else word
            for word in tokens 
            if word not in self.stopword_list or word in self.exclude_stopwords
        ]
        processed_text = " ".join(processed_tokens)
        return re.sub(r'\s+', ' ', processed_text).strip()

    def clean(self):
        text = self.text.lower()
        text = re.sub(r"[^a-zA-Z\s']", ' ', text)
        text = self.process_split_nya(text)
        text = self.correct_typos(text)
        text = self.reduce_extra_characters(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text