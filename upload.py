from huggingface_hub import HfApi

api = HfApi()

# Then upload
api.upload_folder(
    folder_path="data/",
    repo_id="cosita2000/music-recommender-data",
    repo_type="dataset",
)