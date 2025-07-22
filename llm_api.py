from langchain.llms import OpenAI

llm = OpenAI(
    model_name="your-model-name",  # e.g., "gpt-3.5-turbo" or anything you want
    openai_api_key="your-api-key",  # dummy if not checked on server
    openai_api_base="http://localhost:8000/v1",  # your local server address
    temperature=0.7
)

response = llm("What is LangChain?")
print(response)
