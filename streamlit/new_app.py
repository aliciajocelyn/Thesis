import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent  # Thesis/
sys.path.append(str(ROOT))

import yaml
from pathlib import Path
import os
import logging
from datetime import datetime
import streamlit as st

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bertopic import BERTopic

from utils.sentiment_prediction import predict_sentiment
from utils.topic_prediction import prepare_dataset, predict_topics

# ======================
# CONFIG
# ======================
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"  # go up from streamlit/ to Thesis/
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

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

# # ======================
# # FUNCTIONS
# # ======================
# @st.cache_resource
# def load_model():
#     logger.info("Loading models...")

#     # INDOBERT
#     indobert_model_path = str(Path(config["paths"]["indobert_model"]))
#     tokenizer_path = str(Path(config["paths"]["indobert_tokenizer"]))

#     indobert_model = AutoModelForSequenceClassification.from_pretrained(
#         indobert_model_path, local_files_only=True, trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_path, local_files_only=True
#     )

#     # BERTopic
#     embedding_positive = config["embeddings"]["positive"]
#     embedding_negative = config["embeddings"]["negative"]
#     bertopic_positive_model_path = str(Path(config["paths"]["bertopic_positive"]))
#     bertopic_negative_model_path = str(Path(config["paths"]["bertopic_negative"]))

#     bertopic_model_positive = BERTopic.load(bertopic_positive_model_path, embedding_model=embedding_positive)
#     bertopic_model_negative = BERTopic.load(bertopic_negative_model_path, embedding_model=embedding_negative)

#     logger.info("Models loaded successfully.")
#     return indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative

# indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative = load_model()


# @st.cache_resource
# def load_model():
#     logger.info("Loading models...")
#     base_path = Path(__file__).parent / "src" / "models"

#     # INDOBERT
#     indobert_model_path = os.path.abspath(base_path / "indobert_model")
#     tokenizer_path = os.path.abspath(base_path / "indobert_tokenizer")

#     indobert_model = AutoModelForSequenceClassification.from_pretrained(
#         str(indobert_model_path), local_files_only=True, trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         str(tokenizer_path), local_files_only=True
#     )

#     # BERTopic
#     embedding_positive = "indobenchmark/indobert-base-p1"
#     embedding_negative = "indobenchmark/indobert-base-p1"
#     bertopic_positive_model_path = os.path.abspath(base_path / "bertopic" / "best_positive_model")
#     bertopic_negative_model_path = os.path.abspath(base_path / "bertopic" / "best_negative_model")

#     from sentence_transformers import SentenceTransformer
#     embedding_model_positive = SentenceTransformer(embedding_positive)
#     embedding_model_negative = SentenceTransformer(embedding_negative)

#     bertopic_model_positive = BERTopic.load(bertopic_positive_model_path, embedding_model=embedding_model_positive, local=True)
#     bertopic_model_negative = BERTopic.load(bertopic_negative_model_path, embedding_model=embedding_model_negative, local=True)

#     # BERTopic
#     # embedding_positive = "indobenchmark/indobert-base-p1"
#     # embedding_negative = "indobenchmark/indobert-base-p1"
#     # bertopic_positive_model_path = os.path.abspath(base_path / "bertopic" / "best_positive_model")
#     # bertopic_negative_model_path = os.path.abspath(base_path / "bertopic" / "best_negative_model")

#     # from sentence_transformers import SentenceTransformer
#     # embedding_model_positive = SentenceTransformer(embedding_positive)
#     # embedding_model_negative = SentenceTransformer(embedding_negative)

#     # bertopic_model_positive = BERTopic.load(bertopic_positive_model_path, embedding_model=embedding_model_positive, local=True)
#     # bertopic_model_negative = BERTopic.load(bertopic_negative_model_path, embedding_model=embedding_model_negative, local=True)

#     logger.info("Models loaded successfully.")
#     return indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative
#     # return indobert_model, tokenizer



from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import os

@st.cache_resource
def load_model():
    logger.info("Loading models...")
    base_path = Path(__file__).parent / "src" / "models"

#     # INDOBERT
    indobert_model_path = os.path.abspath(base_path / "indobert_model")
    tokenizer_path = os.path.abspath(base_path / "indobert_tokenizer")

    indobert_model = AutoModelForSequenceClassification.from_pretrained(
        str(indobert_model_path), local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )

    # BERTopic
    embedding_positive = "indobenchmark/indobert-base-p1"
    embedding_negative = "indobenchmark/indobert-base-p1"
    bertopic_positive_model_path = os.path.abspath(base_path / "bertopic" / "best_positive_model")
    bertopic_negative_model_path = os.path.abspath(base_path / "bertopic" / "best_negative_model")

    # For SentenceTransformer, we need to modify how we load the base model
    def safe_load_sentence_transformer(model_name):
        # First, load the base model safely
        base_model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        # Then, create a SentenceTransformer with this base model
        return SentenceTransformer(modules=[base_model])

    embedding_model_positive = safe_load_sentence_transformer(embedding_positive)
    embedding_model_negative = safe_load_sentence_transformer(embedding_negative)

    bertopic_model_positive = BERTopic.load(bertopic_positive_model_path, embedding_model=embedding_model_positive)
    bertopic_model_negative = BERTopic.load(bertopic_negative_model_path, embedding_model=embedding_model_negative)

    logger.info("Models loaded successfully.")
    return indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative

indobert_model, tokenizer, bertopic_model_positive, bertopic_model_negative = load_model()
# indobert_model, tokenizer = load_model()