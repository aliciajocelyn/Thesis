import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import os
import logging
import time
import streamlit as st
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textwrap
from collections import Counter
import re
import random

from utils.sentiment_prediction import *
from utils.topic_prediction import *

# ======================
# LOGGING CONFIGURATION
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================
# FUNCTIONS
# ======================
@st.cache_resource
def load_model():
    logger.info("Loading models from Hugging Face Hub...")

    try:
        # ======================
        # SENTIMENT ANALYSIS
        # ======================
        indobert_model = AutoModelForSequenceClassification.from_pretrained(
            "120inblue/sentiment-indobert",   
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "120inblue/sentiment-indobert"
        )
        logger.info("✅ IndoBERT sentiment model loaded.")

        # ======================
        # TOPIC MODELING (POSITIVE)
        # ======================
        embedding_model_positive = SentenceTransformer(
            f"120inblue/topic-positive-embeddings"
        )
        bertopic_model_positive = BERTopic.load(
            f"120inblue/topic-positive-model",
            embedding_model=embedding_model_positive
        )
        positive_topic_names = {
            -1: "Event dan Teman dari Organisasi Kampus (Outlier)",
            0: "Lingkungan Kampus dan Dosen",
            1: "Pertemanan, Relasi dan Koneksi",
            2: "Kegiatan dan Belajar Bersama Teman",
            3: "Program Magang dan Enrichment",
            4: "Kegiatan dan Acara Kampus"
        }
        bertopic_model_positive.set_topic_labels(positive_topic_names)
        logger.info("✅ Positive BERTopic model loaded.")

        # ======================
        # TOPIC MODELING (NEGATIVE)
        # ======================
        embedding_model_negative = SentenceTransformer(
            f"120inblue/topic-negative-embeddings"
        )
        bertopic_model_negative = BERTopic.load(
            f"120inblue/topic-negative-model",
            embedding_model=embedding_model_negative
        )
        negative_topic_names = {
            -1: "Kualitas Fasilitas Kampus dan Pengajaran Dosen (Outlier)",
            0: "Fasilitas Kampus: Wifi, Toilet, dan Infrastruktur",
            1: "Relevansi Materi Kuliah dan Metode Pengajaran Dosen",
            2: "Kebersihan dan Kelengkapan Fasilitas Toilet Kampus",
            3: "Jadwal Kuliah Pagi dan Relevansi Mata Kuliah",
            4: "Beban Tugas Mahasiswa"
        }
        bertopic_model_negative.set_topic_labels(negative_topic_names)
        logger.info("✅ Negative BERTopic model loaded.")

    except Exception as e:
        logger.error(f"❌ Failed to load one of the models: {e}")
        raise

    logger.info("🎉 All models loaded successfully from Hugging Face Hub.")
    return indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative

indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative = load_model()

def plot_top5_words(processed_texts, title, color, width=6, height=4):
    all_words = []
    for text in processed_texts:
        if isinstance(text, str) and text.strip():
            words = [word for word in text.split() if re.match(r'^[a-zA-Z]{2,}$', word)]
            all_words.extend(words)
    
    if not all_words:
        return None
    
    word_counts = Counter(all_words)
    top_5 = word_counts.most_common(5)
    
    if not top_5:
        return None
    
    words, counts = zip(*top_5)
    
    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.barh(words, counts, color=color)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=10)
    
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_ylabel('Words', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis() 
    plt.tight_layout()
    return fig


def generate_wordcloud(texts, title, width=800, height=350, top_k=10):
    combined_text = " ".join(texts)
    if not combined_text.strip():
        return None

    freqs = Counter(combined_text.split())
    top_k_words = set([w for w, _ in freqs.most_common(top_k)])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if word in top_k_words:
            return f"hsl({random.randint(0, 360)}, 100%, 40%)"
        else:
            return f"hsl({random.randint(0, 360)}, 30%, 70%)"

    wc = WordCloud(
        width=width, 
        height=height, 
        background_color="white",
        color_func=color_func
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig


# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="🎓 Student Review Analyzer", layout="wide", initial_sidebar_state="auto")
st.title("🎓 Sentiment Analysis on Student Reviews in Higher Education")
st.caption("Upload reviews → Analyze sentiment → Explore topics")

for key in ["df_labeled", "df_pred_positive", "df_pred_negative", "uploaded_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📊 Sentiment Results", "💡 Topic Results"])

# ==== TAB 1: UPLOAD & ANALYSIS ====
with tab1:
    st.header("Step 1: Upload CSV")
    students_experience = st.file_uploader("Choose a CSV file", type="csv", help="CSV must contain a 'text' column.")

    if students_experience is not None:
        df = pd.read_csv(students_experience)

        if "text" in df.columns and df["text"].dropna().iloc[0] and isinstance(df["text"].dropna().iloc[0], str):
            st.session_state.uploaded_df = df['text']
            logger.info(f"Uploaded file: {students_experience.name}, Rows: {df.shape[0]}")
            st.subheader("📋 Preview of Uploaded Data")
            st.write(f"Uploaded {df.shape[0]} rows")
            st.dataframe(
                df[['text']],
                use_container_width=True, 
                hide_index=True, 
                height=400,
                column_config={
                    "text": st.column_config.TextColumn(
                        "text",
                        help="Student review text, double click to view the full text",
                    )
                }
            )

            if st.button("Run Prediction", icon="🔍"):
                with st.spinner("Predicting student's sentiment..."):
                    try:
                        start_time = time.time()
                        df_labeled = predict_sentiment(df, indobert_model, tokenizer)
                        end_time = time.time()
                        logger.info(f"Sentiment prediction completed in {end_time - start_time:.2f} seconds")
                        st.session_state.df_labeled = df_labeled
                        logger.info("Sentiment prediction completed.")
                        st.success("✅ Sentiment prediction complete! Go to **Sentiment Results Tab** to view results.")
                    except Exception as e:
                        logger.exception("Error during sentiment prediction")
                        st.error(f"Error: {e}")
                if st.session_state.df_labeled is not None:
                    with st.spinner("Predicting topics..."):
                        try:
                            start_time = time.time()
                            texts_pos, texts_neg = prepare_dataset(st.session_state.df_labeled)
                            
                            df_pred_positive = predict_topics(
                                texts=texts_pos, 
                                df=st.session_state.df_labeled,
                                model=bertopic_model_positive, 
                                sentiment_label=1
                            )
                            df_pred_negative = predict_topics(
                                texts=texts_neg, 
                                df=st.session_state.df_labeled,
                                model=bertopic_model_negative, 
                                sentiment_label=0
                            )
                            end_time = time.time()
                            logger.info(f"Topic prediction completed in {end_time - start_time:.2f} seconds")
                            st.session_state.df_pred_positive = df_pred_positive
                            st.session_state.df_pred_negative = df_pred_negative
                            logger.info("Topic prediction completed.")
                            st.success("✅ Topic prediction complete! Go to **Topic Results Tab** to view results.")
                        except Exception as e:
                            logger.exception("Error during topic prediction")
                            st.error(f"Error: {e}")
        else:
            st.error("❗ Make sure the uploaded CSV contains a 'text' column with valid strings.")


# ==== TAB 2: SENTIMENT RESULTS ====
with tab2:
    st.header("Sentiment Distribution")
    if st.session_state.df_labeled is not None:
        df_labeled = st.session_state.df_labeled
        
        if set(df_labeled['sentiment'].unique()).issubset({0, 1}):
            df_labeled['sentiment'] = df_labeled['sentiment'].map({1: 'positive', 0: 'negative'})
        
        df_show = df_labeled[['text', 'sentiment']]
        st.dataframe(
            df_show, 
            use_container_width=True, 
            hide_index=True, 
            height=300,
            column_config={
                "text": st.column_config.TextColumn(
                    "Text",
                    width=370,
                    help="Student review text, double click to view the full text"
                ),
                "sentiment": st.column_config.TextColumn(
                    "Sentiment",
                    width=10
                )
            }
        )
        st.download_button(
            label="💾 Download Labeled Data as CSV",
            data=df_labeled[['text', 'sentiment']].to_csv(index=False),
            file_name="labeled_reviews.csv",
            mime="text/csv",
            key="download_labeled_data_button",
            help="Download the labeled dataframe as CSV"
        )
        
        sentiment_counts = df_labeled["sentiment"].value_counts(normalize=True)*100
        col1, col2, col3 = st.columns([1, 12, 1]) 
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6)) 
            colors = {'positive': '#91DA73', 'negative': '#FF4747'}
            
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                            color=[colors[sentiment] for sentiment in sentiment_counts.index])
            
            for bar, value in zip(bars, sentiment_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylim(0, 100) 
            ax.set_ylabel('Proportion', fontsize=10)
            ax.set_xlabel('Sentiment', fontsize=10)
            ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14, rotation=0)
            ax.tick_params(axis='y', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        df_positive = df_labeled[df_labeled["sentiment"] == 'positive']
        df_negative = df_labeled[df_labeled["sentiment"] == 'negative']

        df_positive['cleaned_text'] = df_positive['text'].apply(lambda x: clean_text(x, negation=False))
        df_positive['processed_text'] = df_positive['cleaned_text'].apply(lambda x: text_preprocessing_sentiment(x, method='sastrawi'))
        
        df_negative['cleaned_text'] = df_negative['text'].apply(lambda x: clean_text(x, negation=False))
        df_negative['processed_text'] = df_negative['cleaned_text'].apply(lambda x: text_preprocessing_sentiment(x, method='sastrawi'))

        st.markdown("<h4 style='text-align: center; color: white;'>Positive Sentiment Analysis</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pos_wc = generate_wordcloud(df_positive["processed_text"], "Word Cloud", width=800, height=500)
            if pos_wc: 
                st.pyplot(pos_wc)
            else: 
                st.info("No positive reviews found.")
        
        with col2:
            pos_top5 = plot_top5_words(df_positive["processed_text"], "Top 5 Words", "#91DA73")
            if pos_top5:
                st.pyplot(pos_top5)
            else:
                st.info("No positive reviews found.")

        st.divider()

        st.markdown("<h4 style='text-align: center; color: white;'>Negative Sentiment Analysis</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            neg_wc = generate_wordcloud(df_negative["processed_text"], "Word Cloud", width=800, height=500)
            if neg_wc:
                st.pyplot(neg_wc)
            else:
                st.info("No negative reviews found.")
        
        with col2:
            neg_top5 = plot_top5_words(df_negative["processed_text"], "Top 5 Words", "#FF4747")
            if neg_top5:
                st.pyplot(neg_top5)
            else:
                st.info("No negative reviews found.")
    else:
        st.warning("⚠ Please run sentiment prediction in Tab 1.")


# ==== TAB 3: TOPIC RESULTS ====
with tab3:
    st.header("Topic Analysis per Sentiment")
    if st.session_state.df_pred_positive is not None and st.session_state.df_pred_negative is not None:
        st.subheader("💬 Positive Sentiment Topics")
        topic_counts_pos = st.session_state.df_pred_positive['topic_name'].value_counts(normalize=True) * 100
        pos_data = topic_counts_pos.reset_index()
        pos_data.columns = ['Topic Name', 'Percentage']

        pos_data['Topic Name Wrapped'] = pos_data['Topic Name'].apply(
            lambda x: "\n".join(textwrap.wrap(x, width=20))
        )

        fig_pos_bar, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(pos_data['Topic Name Wrapped'], pos_data['Percentage'], color="#91DA73")

        for bar, pct in zip(bars, pos_data['Percentage']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=12
            )

        max_height = pos_data['Percentage'].max()
        ax.set_ylim(0, max_height * 1.15)

        ax.set_xlabel("Topic Name", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.set_title("Positive Sentiment Topics", fontsize=16)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=12)

        st.pyplot(fig_pos_bar)

        selected_topic_pos = st.selectbox("Select a Positive Topic", options=topic_counts_pos.index)
        col1, col2 = st.columns(2)
        
        selected_texts_pos = st.session_state.df_pred_positive[
            st.session_state.df_pred_positive['topic_name'] == selected_topic_pos]['text']
        # cleaned_texts_pos = selected_texts_pos.apply(lambda x: clean_text(x, negation=False))
        # processed_texts_pos = cleaned_texts_pos.apply(text_preprocessing_topic)
        processed_texts_pos = selected_texts_pos.apply(lambda x: clean_text_topic_for_wordcloud(x, negation=False))
        
        with col1: 
            fig_pos = generate_wordcloud(
                processed_texts_pos,
                f"Word Cloud for Positive Topic: {selected_topic_pos}",
                width=800,
                height=500

            )
            if fig_pos: st.pyplot(fig_pos) 
        with col2:
            fig_top5_pos = plot_top5_words(processed_texts_pos, f"Top 5 Words for Positive Topic: {selected_topic_pos}", color="#74b8ff")
            if fig_top5_pos: 
                st.pyplot(fig_top5_pos)
        
        st.divider()

        st.subheader("💬 Negative Sentiment Topics")
        topic_counts_neg = st.session_state.df_pred_negative['topic_name'].value_counts(normalize=True) * 100
        neg_data = topic_counts_neg.reset_index()
        neg_data.columns = ['Topic Name', 'Percentage']

        neg_data['Topic Name Wrapped'] = neg_data['Topic Name'].apply(
            lambda x: "\n".join(textwrap.wrap(x, width=20))
        )

        fig_neg_bar, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(neg_data['Topic Name Wrapped'], neg_data['Percentage'], color="#FF4747")

        for bar, pct in zip(bars, neg_data['Percentage']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=12
            )

        max_height = neg_data['Percentage'].max()
        ax.set_ylim(0, max_height * 1.15)

        ax.set_xlabel("Topic Name", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.set_title("Negative Sentiment Topics", fontsize=16)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=12)

        st.pyplot(fig_neg_bar)
        selected_topic_neg = st.selectbox("Select a Negative Topic", options=topic_counts_neg.index)
        col1, col2 = st.columns(2)
        
        selected_texts_neg = st.session_state.df_pred_negative[
            st.session_state.df_pred_negative['topic_name'] == selected_topic_neg]['text']
        # cleaned_texts_neg = selected_texts_neg.apply(lambda x: clean_text(x, negation=True))
        # processed_texts_neg = cleaned_texts_neg.apply(text_preprocessing_topic)
        processed_texts_neg = selected_texts_neg.apply(lambda x: clean_text_topic_for_wordcloud(x, negation=True))
        
        with col1: 
            fig_neg = generate_wordcloud(
                processed_texts_neg,
                f"Word Cloud for Negative Topic: {selected_topic_neg}",
                width=800,
                height=500
            )
            if fig_neg: st.pyplot(fig_neg) 
        with col2:
            fig_top5_neg = plot_top5_words(processed_texts_neg, f"Top 5 Words for Negative Topic: {selected_topic_neg}", color="#74b8ff")
            if fig_top5_neg: 
                st.pyplot(fig_top5_neg)

        st.divider()
        st.markdown("<h4 style='text-align: center; color: white;'>Download All Texts with Sentiment and Topics</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2: 
        if st.session_state.df_pred_positive is not None and st.session_state.df_pred_negative is not None:
            all_df = pd.concat([st.session_state.df_pred_positive, st.session_state.df_pred_negative], axis=0, ignore_index=True)
            all_df["sentiment"] = all_df["sentiment"].map({1: "positive", 0: "negative"})
            relevant_columns = [col for col in ['text', 'sentiment', 'topic_name'] if col in all_df.columns]
            if relevant_columns:
                download_df = all_df[relevant_columns]
            else:
                download_df = all_df
            csv_bytes = download_df.to_csv(index=False).encode('utf-8')
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                st.download_button(
                    label="📥 Download All Results as CSV",
                    data=csv_bytes,
                    file_name="predicted_results.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠ Please run topic prediction in Tab 1.")
