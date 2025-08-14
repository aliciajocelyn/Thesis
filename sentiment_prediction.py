import re
import pandas as pd
import numpy as np
import pickle

import torch
from src.dictionary.normalization_dictionary import norm_dict

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
exception_words = ["tanya", 'punya', 'bertanya', 'hanya']
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
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = process_split_nya(text)
    text = correct_typos(text)
    text = reduce_extra_characters(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Label mapping
id2label = {0: "Negative", 1: "Positive"}

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