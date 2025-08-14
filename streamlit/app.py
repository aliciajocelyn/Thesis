from pathlib import Path
import os
import logging
from datetime import datetime
import streamlit as st
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bertopic import BERTopic

from utils.sentiment_prediction import predict_sentiment
from utils.topic_prediction import prepare_dataset, predict_topics

from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    logger.info("Loading models...")
    base_path = Path(__file__).parent / "src" / "models"
    indobert_model_path = base_path / "indobert_model"
    tokenizer_path = base_path / "indobert_tokenizer"
    bertopic_positive_model_path = base_path / "bertopic" / "best_positive_model"
    bertopic_negative_model_path = base_path / "bertopic" / "best_negative_model"

    indobert_model = AutoModelForSequenceClassification.from_pretrained(
        str(indobert_model_path), local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )

    bertopic_model_positive = BERTopic.load(bertopic_positive_model_path)
    bertopic_model_negative = BERTopic.load(bertopic_negative_model_path)

    logger.info("Models loaded successfully.")
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
            st.session_state.uploaded_df = df
            logger.info(f"Uploaded file: {students_experience.name}, Rows: {df.shape[0]}")
            st.subheader("📋 Preview of Uploaded Data")
            st.write(df.head())

            if st.button("🔍 Run Sentiment Prediction"):
                with st.spinner("Predicting sentiment..."):
                    try:
                        sample_df = df.sample(n=min(10, len(df)), random_state=42)
                        df_labeled = predict_sentiment(sample_df, indobert_model, tokenizer)
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
                            texts=texts_pos, df=st.session_state.df_labeled,
                            model=bertopic_model_positive, sentiment_label=1
                        )
                        df_pred_negative = predict_topics(
                            texts=texts_neg, df=st.session_state.df_labeled,
                            model=bertopic_model_negative, sentiment_label=0
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
        st.write(df_labeled.head(10))
        sentiment_counts = df_labeled["sentiment"].value_counts(normalize=True) * 100
        st.bar_chart(sentiment_counts)

        col1, col2 = st.columns(2)
        with col1:
            pos_fig = generate_wordcloud(df_labeled[df_labeled["sentiment"] == 1]["text"], "Positive Reviews Word Cloud")
            if pos_fig: st.pyplot(pos_fig)
            else: st.info("No positive reviews found.")

        with col2:
            neg_fig = generate_wordcloud(df_labeled[df_labeled["sentiment"] == 0]["text"], "Negative Reviews Word Cloud")
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
        topic_counts_pos = st.session_state.df_pred_positive['topic_name'].value_counts()
        st.bar_chart(topic_counts_pos)
        selected_topic_pos = st.selectbox("Select a Positive Topic", options=topic_counts_pos.index)
        fig_pos = generate_wordcloud(
            st.session_state.df_pred_positive[st.session_state.df_pred_positive['topic_name'] == selected_topic_pos]['text'],
            f"Word Cloud for Positive Topic: {selected_topic_pos}"
        )
        if fig_pos: st.pyplot(fig_pos)

        st.divider()

        # Negative topics
        st.subheader("💬 Negative Sentiment Topics")
        topic_counts_neg = st.session_state.df_pred_negative['topic_name'].value_counts()
        st.bar_chart(topic_counts_neg)
        selected_topic_neg = st.selectbox("Select a Negative Topic", options=topic_counts_neg.index)
        fig_neg = generate_wordcloud(
            st.session_state.df_pred_negative[st.session_state.df_pred_negative['topic_name'] == selected_topic_neg]['text'],
            f"Word Cloud for Negative Topic: {selected_topic_neg}"
        )
        if fig_neg: st.pyplot(fig_neg)
    else:
        st.warning("⚠ Please run topic prediction in Tab 1.")




# from pathlib import Path
# import os
# import streamlit as st
# import pandas as pd

# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from bertopic import BERTopic

# from sentiment_prediction import predict_sentiment
# from topic_prediction import prepare_dataset, predict_topics

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # FUNCTIONS
# @st.cache_resource
# def load_model():
#     base_path = Path(__file__).parent / "src" / "models"
#     indobert_model_path = base_path / "indobert_model"
#     tokenizer_path = base_path / "indobert_tokenizer"
#     bertopic_positive_model_path = base_path / "bertopic" / "best_positive_model"
#     bertopic_negative_model_path = base_path / "bertopic" / "best_negative_model"

#     indobert_model = AutoModelForSequenceClassification.from_pretrained(
#         str(indobert_model_path), local_files_only=True, trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         str(tokenizer_path), local_files_only=True
#     )

#     bertopic_model_positive = BERTopic.load(bertopic_positive_model_path)
#     bertopic_model_negative = BERTopic.load(bertopic_negative_model_path)

#     return indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative

# indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative = load_model()


# def plot_wordcloud_for_topic(df, topic_name, text_column="text"):
#     text_data = " ".join(df[df['topic_name'] == topic_name][text_column].dropna())
#     if not text_data.strip():
#         return None
#     wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     ax.set_title(f"Word Cloud for Topic: {topic_name}")
#     return fig

# def generate_wordcloud(texts, title):
#     """Generate a matplotlib figure for a given list of texts."""
#     combined_text = " ".join(texts)
#     if not combined_text.strip():
#         return None
#     wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation="bilinear")
#     ax.axis("off")
#     ax.set_title(title)
#     return fig


# ########################################################################################################
# # st.title("🎓 Sentiment Analysis on Student Reviews in Higher Education")
# # st.write("📁 Upload a CSV file with a column named `text` containing student reviews:")

# st.set_page_config(page_title="🎓 Student Review Analyzer", layout="wide")
# st.title("🎓 Sentiment Analysis on Student Reviews in Higher Education")
# st.caption("Upload reviews → Analyze sentiment → Explore topics")

# if "df_labeled" not in st.session_state:
#     st.session_state.df_labeled = None
# if "df_pred_positive" not in st.session_state:
#     st.session_state.df_pred_positive = None
# if "df_pred_negative" not in st.session_state:
#     st.session_state.df_pred_negative = None

# tab1, tab2, tab3 = st.tabs(["📁 Upload Data", "📊 Sentiment Results", "💡 Topic Results"])

# # ==== TAB 1: UPLOAD DATA ====
# with tab1:
#     st.header("Step 1: Upload CSV")
#     st.write("Your file must contain a column named `text` with student reviews.")
#     students_experience = st.file_uploader("Choose a CSV file", type="csv")

#     if students_experience is not None:
#         df = pd.read_csv(students_experience)

#         if "text" in df.columns and df["text"].dropna().iloc[0] and isinstance(df["text"].dropna().iloc[0], str):
#             st.subheader("📋 Preview of Uploaded Data")
#             st.write(df.head())

#             if st.button("🔍 Run Sentiment & Topic Prediction"):
#                 with st.spinner("Analyzing... This may take a moment ⏳"):
#                     df_labeled = predict_sentiment(df, indobert_model, tokenizer)
#                     st.session_state.df_labeled = df_labeled

#                     # Prepare topic inputs
#                     texts_pos, texts_neg = prepare_dataset(df_labeled)

#                     # Predict topics separately
#                     df_pred_positive = predict_topics(
#                         texts=texts_pos, df=df_labeled, model=bertopic_model_positive, sentiment_label=1
#                     )
#                     df_pred_negative = predict_topics(
#                         texts=texts_neg, df=df_labeled, model=bertopic_model_negative, sentiment_label=0
#                     )

#                     st.session_state.df_pred_positive = df_pred_positive
#                     st.session_state.df_pred_negative = df_pred_negative

#                 st.success("✅ Analysis complete! Go to the other tabs to view results.")
#         else:
#             st.error("❗ Make sure the uploaded CSV contains a 'text' column with valid strings.")



# # ==== TAB 2: SENTIMENT RESULTS ====
# with tab2:
#     st.header("Sentiment Distribution")
#     if st.session_state.df_labeled is not None:
#         df_labeled = st.session_state.df_labeled

#         # Sentiment bar chart
#         sentiment_counts = df_labeled["sentiment"].value_counts(normalize=True) * 100
#         st.bar_chart(sentiment_counts)

#         # Word clouds for each sentiment
#         col1, col2 = st.columns(2)

#         with col1:
#             pos_fig = generate_wordcloud(
#                 df_labeled[df_labeled["sentiment"] == "positive"]["text"],
#                 "Positive Reviews Word Cloud"
#             )
#             if pos_fig:
#                 st.pyplot(pos_fig)
#             else:
#                 st.info("No positive reviews found.")

#         with col2:
#             neg_fig = generate_wordcloud(
#                 df_labeled[df_labeled["sentiment"] == "negative"]["text"],
#                 "Negative Reviews Word Cloud"
#             )
#             if neg_fig:
#                 st.pyplot(neg_fig)
#             else:
#                 st.info("No negative reviews found.")

#         st.divider()

#         # Data view button
#         if st.button("View data"):
#             num = st.number_input(
#                 "How many rows do you want to see?",
#                 min_value=1,
#                 max_value=len(df_labeled),
#                 value=5,
#                 step=1
#             )
#             st.write(f"📋 Showing first {num} labeled reviews")
#             st.data_editor(
#                 df_labeled[["text", "sentiment"]].head(num),
#                 num_rows="dynamic",
#                 use_container_width=True
#             )
#     else:
#         st.warning("⚠ Please upload and analyze data in the first tab.")



# # ==== TAB 3: TOPIC RESULTS ====
# with tab3:
#     st.header("Topic Analysis per Sentiment")
#     if st.session_state.df_pred_positive is not None and st.session_state.df_pred_negative is not None:

#         # === Positive sentiment topics ===
#         st.subheader("💬 Positive Sentiment Topics")
#         topic_counts_pos = st.session_state.df_pred_positive['topic_name'].value_counts()
#         st.bar_chart(topic_counts_pos)

#         selected_topic_pos = st.selectbox(
#             "Select a Positive Topic to View Word Cloud",
#             options=topic_counts_pos.index
#         )
#         fig_pos = generate_wordcloud(
#             st.session_state.df_pred_positive[
#                 st.session_state.df_pred_positive['topic_name'] == selected_topic_pos
#             ]['text'],
#             f"Word Cloud for Positive Topic: {selected_topic_pos}"
#         )
#         if fig_pos:
#             st.pyplot(fig_pos)
#         else:
#             st.info("No words found for this topic.")

#         st.divider()

#         # === Negative sentiment topics ===
#         st.subheader("💬 Negative Sentiment Topics")
#         topic_counts_neg = st.session_state.df_pred_negative['topic_name'].value_counts()
#         st.bar_chart(topic_counts_neg)

#         selected_topic_neg = st.selectbox(
#             "Select a Negative Topic to View Word Cloud",
#             options=topic_counts_neg.index
#         )
#         fig_neg = generate_wordcloud(
#             st.session_state.df_pred_negative[
#                 st.session_state.df_pred_negative['topic_name'] == selected_topic_neg
#             ]['text'],
#             f"Word Cloud for Negative Topic: {selected_topic_neg}"
#         )
#         if fig_neg:
#             st.pyplot(fig_neg)
#         else:
#             st.info("No words found for this topic.")

#         st.divider()

#         # === BERTopic visualizations ===
#         st.subheader("🔍 Topic Clusters & Similarity (Interactive)")
#         st.write("Positive Sentiment Topic Map")
#         st.plotly_chart(st.session_state.bertopic_model_positive.visualize_topics(), use_container_width=True)

#         st.write("Negative Sentiment Topic Map")
#         st.plotly_chart(st.session_state.bertopic_model_negative.visualize_topics(), use_container_width=True)

#     else:
#         st.warning("⚠ Please upload and analyze data in the first tab.")



# # students_experience = st.file_uploader("Choose a CSV file", type="csv")

# # if students_experience is not None:
# #     df = pd.read_csv(students_experience)

# #     if 'text' in df.columns and df['text'].dropna().iloc[0] is not None and isinstance(df['text'].dropna().iloc[0], str):
# #         st.subheader("Sample of Uploaded Reviews")
# #         st.write(df['text'].head(5))

# #         if st.button("Predict Sentiment", icon='🔍'):
# #             with st.spinner("Processing and predicting..."):
# #                 df_labeled = predict_sentiment(df, indobert_model, tokenizer)

# #                 # Show distribution
# #                 sentiment_counts = df_labeled['sentiment'].value_counts(normalize=True) * 100
# #                 st.subheader("📊 Sentiment Distribution")
# #                 st.bar_chart(sentiment_counts)

# #                 st.subheader("🔎 Preview of Labeled Data")
# #                 st.write(df_labeled[['text', 'sentiment']].head(10))

# #                 texts_pos, texts_neg = prepare_dataset(df_labeled)
# #                 df_pred_positive = predict_topics(
# #                     texts=texts_pos, 
# #                     df=df_labeled, 
# #                     model=bertopic_model_positive, 
# #                     sentiment_label='positive'    
# #                 )
                
# #                 df_pred_negative = predict_topics(
# #                     texts=texts_neg, 
# #                     df=df_labeled, 
# #                     model=bertopic_model_negative, 
# #                     sentiment_label='negative'    
# #                 )




# #     else:
# #         st.error("❗ Make sure the uploaded CSV contains a 'text' column with valid strings.")

# # if __name__ == "__main__":
# #     load_model()
