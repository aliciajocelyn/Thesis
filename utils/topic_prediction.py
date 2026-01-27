import re
from src.dictionary.exclude_words import exclude_stopwords
from src.models.preprocess.text_cleansing import TextCleansing

def clean_text(text, negation=False):
    tc = TextCleansing(text, exclude_stopwords)
    return tc.clean_text_topic(negation)

def text_preprocessing_topic(text):
    tc = TextCleansing(text, exclude_stopwords)
    return tc.text_preprocessing_topic()

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

from src.dictionary.topic_name_mapping import POSITIVE_TOPIC_MAPPING, NEGATIVE_TOPIC_MAPPING
def map_topic_names(topic_name, sentiment_label):
    if sentiment_label == 1:  
        return POSITIVE_TOPIC_MAPPING.get(topic_name, topic_name)
    else:  
        return NEGATIVE_TOPIC_MAPPING.get(topic_name, topic_name)

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
    
    df_pred['topic_name'] = df_pred['topic_name'].apply(
        lambda x: map_topic_names(x, sentiment_label)
    )

    return df_pred


def clean_text_topic_for_wordcloud(text, negation=False):
    text = clean_text(text, negation=negation)
    text = text_preprocessing_topic(text)

    return text
