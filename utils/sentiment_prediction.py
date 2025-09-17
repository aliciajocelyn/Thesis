import re
import pandas as pd
import numpy as np

from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import torch
from src.dictionary.exclude_words import exclude_stopwords
from src.dictionary.normalization_dictionary import norm_dict

# Remove Stopwords
# nlp_id
stopword = StopWord()
tokenizer = Tokenizer()
custom_stopwords = ['nya', 'ya', 'nih', 'jadi', 'ga']
nlp_stop_words = stopword.get_stopword()
nlp_stop_words.append(custom_stopwords)

# sastrawi
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())
custom_stopwords = {'nya', 'ya', 'nih', 'jadi', 'ga'}
sastrawi_stop_words = stopwords.union(custom_stopwords)

exception_words = ["tanya", 'punya', 'bertanya', 'hanya'] # Kata-kata yang berakhiran dengan -nya namun memiliki arti sendiri

# Typo correction
def correct_typos(text):
    for typo, correction in norm_dict.items():
        text = re.sub(typo, correction, text, flags=re.IGNORECASE)
    return text

# Reduce repeated characters
def reduce_extra_characters(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

# Handle "-nya" suffix
def split_nya(word):
    if word in exception_words:
        return word
    return re.sub(r'(.*?)nya$', r'\1 nya', word)

def process_split_nya(review):
    words = review.split()
    processed_words = [split_nya(word.strip()) for word in words]
    return ' '.join(processed_words)

# Clean text
def text_cleansing(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = process_split_nya(text)
    text = correct_typos(text)
    text = reduce_extra_characters(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_preprocessing_sentiment(text, method):
    if method == 'nlp_id':
        # Menambahkan kata untuk stop words
        tokens = tokenizer.tokenize(text)
        filtered_tokens = [
            word for word in tokens 
            if (word not in nlp_stop_words or word in exclude_stopwords) and word != "tidak"
        ]


        text = re.sub(r'\s+', ' ', text).strip()

        processed_text = " ".join(filtered_tokens)

        return processed_text
    
    if method == 'sastrawi':
        tokens = text.lower().split()

        # Filter stopword
        filtered_tokens = [
            word for word in tokens 
            if (word not in sastrawi_stop_words or word in exclude_stopwords) and word != "tidak"
        ]

        # Gabung kembali & rapikan spasi
        cleaned_text = re.sub(r'\s+', ' ', ' '.join(filtered_tokens)).strip()

        return cleaned_text

# Label mapping
id2label = {0: "negative", 1: "positive"}

# Inference function
def run_indobert_pipeline(df, model, tokenizer):
    model.eval()

    texts = df['cleaned_text'].tolist()
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        labels = [id2label[int(p)] for p in preds]

    return labels

def predict_sentiment(df, model, tokenizer): 
    df['cleaned_text'] = df['text'].apply(text_cleansing)
    predictions = run_indobert_pipeline(df, model, tokenizer)
    df['sentiment'] = predictions

    return df