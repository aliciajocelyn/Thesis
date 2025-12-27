import re
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord

class NLPIdPreprocessor: 
    def __init__(self, exclude_lemmatization, exclude_stopwords, add_stopwords=['nya', 'ya', 'nih']):
        self.exclude_lemmatization = exclude_lemmatization
        self.exclude_stopwords = exclude_stopwords

        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()
        self.stopword = StopWord()

        self.stopword_list = self.stopword.get_stopword()
        self.stopword_list.append(add_stopwords)

    def transform(self, text: str) -> str: 
        tokens = self.tokenizer.tokenize(text)

        processsed_tokens = [
            self.lemmatizer.lemmatize(word) if (word not in self.exclude_stopwords and word not in self.exclude_lemmatization) else word
            for word in tokens
            if word not in self.stopword_list or word in self.exclude_stopwords
        ]
        return re.sub(r'\s+', ' ', ' '.join(processsed_tokens)).strip()