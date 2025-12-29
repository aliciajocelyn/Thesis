from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Authenticate
api = HfApi()
username = api.whoami()["name"]

# -------- Sentiment Repo --------
sentiment_repo_id = f"{username}/sentiment-indobert"
create_repo(repo_id=sentiment_repo_id, exist_ok=True)

# Your merged folder containing BOTH indobert_model + indobert_tokenizer
upload_folder(
    folder_path="src/models/final_indobert",  # <-- merge model+tokenizer here
    repo_id=sentiment_repo_id,
    path_in_repo=".",   # upload files to root of repo
    repo_type="model"
)


# -------- Topic Positive Repo --------
topic_pos_repo_id = f"{username}/topic-positive-model"
create_repo(repo_id=topic_pos_repo_id, exist_ok=True)

upload_folder(
    folder_path="src/models/final_bertopic/positive/best_model", 
    repo_id=topic_pos_repo_id,
    path_in_repo=".",   # keep subfolders best_model, embedding_model
    repo_type="model"
)

# -------- Topic Positive Repo --------
topic_pos_repo_id = f"{username}/topic-positive-embeddings"
create_repo(repo_id=topic_pos_repo_id, exist_ok=True)

upload_folder(
    folder_path="src/models/final_bertopic/positive/embedding_model", 
    repo_id=topic_pos_repo_id,
    path_in_repo=".",   # keep subfolders best_model, embedding_model
    repo_type="model"
)


# -------- Topic Negative Repo --------
topic_neg_repo_id = f"{username}/topic-negative-model"
create_repo(repo_id=topic_neg_repo_id, exist_ok=True)

upload_folder(
    folder_path="src/models/final_bertopic/negative/best_model", 
    repo_id=topic_neg_repo_id,
    path_in_repo=".",
    repo_type="model"
)

# -------- Topic Negative Repo --------
topic_neg_repo_id = f"{username}/topic-negative-embeddings"
create_repo(repo_id=topic_neg_repo_id, exist_ok=True)

upload_folder(
    folder_path="src/models/final_bertopic/negative/embedding_model", 
    repo_id=topic_neg_repo_id,
    path_in_repo=".",
    repo_type="model"
)

print("✅ All models uploaded to Hugging Face Hub!")
