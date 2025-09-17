import re
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord

from src.dictionary.exclude_words import exclude_stopwords

stopword = StopWord()
tokenizer = Tokenizer()

def clean_text(text, negation=True):
    if negation:
        for phrase in ['sangat tidak menyukai', 'tidak menyukai','sangat tidak suka', 'tidak suka', 'kurang suka', 'kurang menyukai', 'ga suka', 'gak suka', 'ga menyukai', 'gak menyukai']:
            text = text.replace(phrase, '')

    for phrase in ['suka', 'sangat suka', 'menyukai', 'sangat menyukai']:
        text = text.replace(phrase, '')

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_preprocessing_topic(text):
    # Menambahkan kata untuk stop words
    stop_words = stopword.get_stopword()
    stop_words.append(exclude_stopwords)

    tokens = tokenizer.tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

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

# Topic name mappings
POSITIVE_TOPIC_MAPPING = {
    "0_teman_kampus_dosen_materi": "Lingkungan Kampus & Dosen",
    "2_bareng_belajar_teman_ujian": "Kegiatan dan Belajar Bersama Teman", 
    "1_bertemu_teman_orang_koneksi": "Pertemanan, Relasi dan Koneksi",
    "-1_event_teman_organisasi_kampus": "Event dan Teman dari Organisasi Kampus",
    "4_kegiatan_program_acara_seru": "Kegiatan & Acara Kampus",
    "3_program_mahasiswa_magang_enrichment": "Program Magang & Enrichment"
}

NEGATIVE_TOPIC_MAPPING = {
    "0_kampus_wifi_fasilitas_toilet": "Fasilitas Kampus: Wifi, Toilet, dan Infrastruktur",
    "1_dosen_materi_mahasiswa_mengajar": "Relevansi Materi Kuliah dan Metode Pengajaran Dosen",
    "-1_fasilitas_kampus_dosen_kuliah": "Kualitas Fasilitas Kampus dan Pengajaran Dosen",
    "2_toilet_tissue_fasilitas_tangan": "Kebersihan dan Kelengkapan Fasilitas Toilet Kampus",
    "3_kuliah_mata_pagi_jam": "Jadwal Kuliah Pagi dan Relevansi Mata Kuliah",
    "4_tugas_ubah_menumpuk_bersamaan": "Beban Tugas Mahasiswa"
}

def map_topic_names(topic_name, sentiment_label):
    """Map auto-generated topic names to custom names based on sentiment"""
    if sentiment_label == 1:  # Positive sentiment
        return POSITIVE_TOPIC_MAPPING.get(topic_name, topic_name)
    else:  # Negative sentiment  
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
    
    # Apply custom topic name mapping
    df_pred['topic_name'] = df_pred['topic_name'].apply(
        lambda x: map_topic_names(x, sentiment_label)
    )

    return df_pred