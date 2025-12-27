import pandas as pd
import numpy as np
import torch

from src.dictionary.normalization_dictionary import norm_dict
from src.models.preprocess.text_cleansing import TextCleansing

# Label mapping
id2label = {0: "negative", 1: "positive"}

def text_cleansing(text):
    tc = TextCleansing(text, norm_dict=norm_dict)
    return tc.clean()

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