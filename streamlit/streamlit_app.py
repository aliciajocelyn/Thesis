import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import os
import logging
from datetime import datetime
import streamlit as st
import pandas as pd

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textwrap
import io

from utils.sentiment_prediction import *
from utils.topic_prediction import *

# ======================
# LOGGING CONFIGURATION
# ======================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
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
            "120inblue/sentiment-indobert",   # replace with your HF repo
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

def generate_wordcloud(texts, title):
    combined_text = " ".join(texts)
    if not combined_text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="🎓 Student Review Analyzer", layout="wide")
st.title("🎓 Sentiment Analysis on Student Reviews in Higher Education")
st.caption("Upload reviews → Analyze sentiment → Explore topics")

# State initialization
for key in ["df_labeled", "df_pred_positive", "df_pred_negative", "uploaded_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📊 Sentiment Results", "💡 Topic Results"])

# ==== TAB 1: UPLOAD & ANALYSIS ====
with tab1:
    st.header("Step 1: Upload CSV")
    students_experience = st.file_uploader("Choose a CSV file", type="csv")

    if students_experience is not None:
        df = pd.read_csv(students_experience)

        if "text" in df.columns and df["text"].dropna().iloc[0] and isinstance(df["text"].dropna().iloc[0], str):
            st.session_state.uploaded_df = df['text']
            logger.info(f"Uploaded file: {students_experience.name}, Rows: {df.shape[0]}")
            st.subheader("📋 Preview of Uploaded Data")
            st.write(df['text'].head())

            if st.button("🔍 Run Sentiment Prediction"):
                with st.spinner("Predicting sentiment..."):
                    try:
                        # sample_df = df.sample(n=min(10, len(df)), random_state=42)
                        df_labeled = predict_sentiment(df, indobert_model, tokenizer)
                        st.session_state.df_labeled = df_labeled
                        logger.info("Sentiment prediction completed.")
                        st.success("✅ Sentiment prediction complete! Go to **Tab 2** to view results.")
                    except Exception as e:
                        logger.exception("Error during sentiment prediction")
                        st.error(f"Error: {e}")

            if st.session_state.df_labeled is not None and st.button("💡 Run Topic Prediction"):
                with st.spinner("Running topic modeling..."):
                    try:
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

                        st.session_state.df_pred_positive = df_pred_positive
                        st.session_state.df_pred_negative = df_pred_negative
                        logger.info("Topic prediction completed.")
                        st.success("✅ Topic prediction complete! Go to **Tab 3** to view results.")
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
        st.write(df_show.head(10))
        sentiment_counts = df_labeled["sentiment"].value_counts(normalize=True)*100
        
        # Create a container to control chart width
        col1, col2, col3 = st.columns([1, 9, 1])  # This creates a centered chart taking 2/4 of the width
        
        with col2:
            # Create a custom bar chart with matplotlib
            fig, ax = plt.subplots(figsize=(10, 4.5))  # Reduced figure size since it's in a smaller container
            
            # Define colors for sentiments
            colors = {'positive': '#91DA73', 'negative': '#FF4747'}  # Green for positive, red for negative
            
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                         color=[colors[sentiment] for sentiment in sentiment_counts.index])
            
            # Add percentage labels on top of bars
            for bar, value in zip(bars, sentiment_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Customize the chart
            ax.set_ylim(0, 100) 
            ax.set_ylabel('Proportion', fontsize=12)
            ax.set_xlabel('Sentiment', fontsize=12)
            ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            
            # Make x-axis labels horizontal and larger
            ax.tick_params(axis='x', labelsize=14, rotation=0)
            ax.tick_params(axis='y', labelsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)

        df_positive = df_labeled[df_labeled["sentiment"] == 'positive']
        df_negative = df_labeled[df_labeled["sentiment"] == 'negative']

        col1, col2 = st.columns(2)
        with col1:
            df_positive['cleaned_text'] = df_positive['text'].apply(lambda x: clean_text(x, negation=False))
            df_positive['processed_text'] = df_positive['cleaned_text'].apply(lambda x: text_preprocessing_sentiment(x, method='sastrawi'))
            pos_fig = generate_wordcloud(df_positive["processed_text"], "Positive Reviews Word Cloud")
            if pos_fig: st.pyplot(pos_fig)
            else: st.info("No positive reviews found.")

        with col2:
            df_negative['cleaned_text'] = df_negative['text'].apply(lambda x: clean_text(x, negation=False))
            df_negative['processed_text'] = df_negative['cleaned_text'].apply(lambda x: text_preprocessing_sentiment(x, method='sastrawi'))
            neg_fig = generate_wordcloud(df_negative["processed_text"], "Negative Reviews Word Cloud")
            if neg_fig: st.pyplot(neg_fig)
            else: st.info("No negative reviews found.")
    else:
        st.warning("⚠ Please run sentiment prediction in Tab 1.")


# ==== TAB 3: TOPIC RESULTS ====
with tab3:
    st.header("Topic Analysis per Sentiment")
    if st.session_state.df_pred_positive is not None and st.session_state.df_pred_negative is not None:
        # Positive topics
        st.subheader("💬 Positive Sentiment Topics")
        topic_counts_pos = st.session_state.df_pred_positive['topic_name'].value_counts(normalize=True) * 100  # percentage
        pos_data = topic_counts_pos.reset_index()
        pos_data.columns = ['Topic Name', 'Percentage']

        # Wrap labels (2 rows max using \n)
        pos_data['Topic Name Wrapped'] = pos_data['Topic Name'].apply(
            lambda x: "\n".join(textwrap.wrap(x, width=20))  # adjust width as needed
        )

        fig_pos_bar, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(pos_data['Topic Name Wrapped'], pos_data['Percentage'], color="#91DA73")

        # Add percentage labels above bars
        for bar, pct in zip(bars, pos_data['Percentage']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,  # a little above the bar
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=12
            )

        # Adjust y-limit so labels fit
        max_height = pos_data['Percentage'].max()
        ax.set_ylim(0, max_height * 1.15)  # 15% extra margin

        ax.set_xlabel("Topic Name", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.set_title("Positive Sentiment Topics", fontsize=16)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=12)

        col1, col2, col3 = st.columns([1,2,1])  # middle column is smaller
        with col2:
            st.pyplot(fig_pos_bar)

        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            selected_topic_pos = st.selectbox("Select a Positive Topic", options=topic_counts_pos.index)
            selected_texts_pos = st.session_state.df_pred_positive[
                st.session_state.df_pred_positive['topic_name'] == selected_topic_pos]['text']
            processed_texts_pos = selected_texts_pos.apply(lambda x: text_preprocessing_topic(clean_text(x, negation=False)))
        
            fig_pos = generate_wordcloud(
                processed_texts_pos,
                f"Word Cloud for Positive Topic: {selected_topic_pos}"
            )
            if fig_pos: st.pyplot(fig_pos) 

        st.divider()

        # Negative topics
        st.subheader("💬 Negative Sentiment Topics")
        topic_counts_neg = st.session_state.df_pred_negative['topic_name'].value_counts(normalize=True) * 100
        neg_data = topic_counts_neg.reset_index()
        neg_data.columns = ['Topic Name', 'Percentage']

        # Wrap labels
        neg_data['Topic Name Wrapped'] = neg_data['Topic Name'].apply(
            lambda x: "\n".join(textwrap.wrap(x, width=20))
        )

        fig_neg_bar, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(neg_data['Topic Name Wrapped'], neg_data['Percentage'], color="#FF4747")

        # Add percentage labels
        for bar, pct in zip(bars, neg_data['Percentage']):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{pct:.1f}%", ha='center', va='bottom', fontsize=12
            )

        # Adjust y-limit so labels fit
        max_height = neg_data['Percentage'].max()
        ax.set_ylim(0, max_height * 1.15)  # 15% extra margin

        ax.set_xlabel("Topic Name", fontsize=14)
        ax.set_ylabel("Percentage (%)", fontsize=14)
        ax.set_title("Negative Sentiment Topics", fontsize=16)
        ax.tick_params(axis='x', rotation=0, labelsize=10)
        ax.tick_params(axis='y', labelsize=12)

        col1, col2, col3 = st.columns([1,2,1])  
        with col2:
            st.pyplot(fig_neg_bar)


        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:
            selected_topic_neg = st.selectbox("Select a Negative Topic", options=topic_counts_neg.index)
            selected_texts_neg = st.session_state.df_pred_negative[
                st.session_state.df_pred_negative['topic_name'] == selected_topic_neg]['text']
            processed_texts_neg = selected_texts_neg.apply(lambda x: text_preprocessing_topic(clean_text(x, negation=True)))
            fig_neg = generate_wordcloud(
                processed_texts_neg,    
                f"Word Cloud for Negative Topic: {selected_topic_neg}"
            )
            if fig_neg: st.pyplot(fig_neg)
    else:
        st.warning("⚠ Please run topic prediction in Tab 1.")
