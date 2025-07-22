import os
from langchain_openai import OpenAIEmbeddings

def get_emb_model():
    embeddings = OpenAIEmbeddings(
        model=os.environ["EMB_MODEL_NAME"],
        api_key=os.environ["EMB_API_KEY"],
        base_url=os.environ["EMB_API_BASE"],
    )
    return embeddings
