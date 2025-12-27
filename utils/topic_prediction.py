import re
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord

from src.dictionary.exclude_words import exclude_stopwords

stopword = StopWord()
tokenizer = Tokenizer()

negation_phrases = [
    'sangat tidak menyukai', 'tidak menyukai', 'sangat tidak suka',
    'tidak suka', 'kurang suka', 'kurang menyukai', 'ga suka',
    'gak suka', 'ga menyukai', 'gak menyukai', 'gada'
]

positive_phrases = ['suka', 'sangat suka', 'menyukai', 'sangat menyukai']

def clean_text(text, negation=True):
    if negation:
        for phrase in negation_phrases:
            text = text.replace(phrase, '')

    for phrase in positive_phrases:
        text = text.replace(phrase, '')

    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = re.sub(r'\bga\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_preprocessing_topic(text): 
    stopwords = stopword.get_stopword()
    stopwords.append(exclude_stopwords)
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    text = re.sub(r'\s+', ' ', text).strip()
    processed_text = " ".join(filtered_tokens)
    return processed_text


def prepare_dataset(df_modeling):
    df_modeling['sentiment'] = df_modeling['sentiment'].map({'negative': 0, 'positive': 1})
    df_pos = df_modeling[df_modeling['sentiment'] == 1].copy()
    df_neg = df_modeling[df_modeling['sentiment'] == 0].copy()

    df_pos['cleaned_text'] = df_pos['cleaned_text'].apply(clean_text, negation=False)
    df_neg['cleaned_text'] = df_neg['cleaned_text'].apply(clean_text, negation=True)

    df_pos['processed_text'] = df_pos['cleaned_text'].apply(text_preprocessing_topic)
    df_neg['processed_text'] = df_neg['cleaned_text'].apply(text_preprocessing_topic)

    text_pos = df_pos['processed_text'].astype(str).tolist()
    text_neg = df_neg['processed_text'].astype(str).tolist()

    return text_pos, text_neg


def predict_topics(texts, df, model, sentiment_label):
    topics, probs = model.transform(texts)

    df_pred = df[df['sentiment'] == sentiment_label].copy()
    df_pred['topic'] = topics
    df_pred['topic_proba'] = probs

    topic_info = model.get_topic_info()
    df_pred = df_pred.merge(
        topic_info[['Topic', 'Name']],
        left_on='topic',
        right_on='Topic',
        how='left'
    ).drop(columns=["Topic"]).rename(columns={"Name": "topic_name"})

    return df_pred