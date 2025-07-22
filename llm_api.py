from langchain.chat_models import init_chat_model
import os
def get_model():
    model = init_chat_model(
        model=os.environ.get("LLM_MODEL_NAME"),  # Can be any name; just consistent with your server
        model_provider=os.environ.get("LLM_MODEL_PROVIDER"),
        base_url=os.environ.get("LLM_API_BASE"),  # your local API endpoint
        openai_api_key=os.environ.get("LLM_API_KEY"),  # dummy if your server doesn’t check it
        temperature=0.3,
    )
    return model


