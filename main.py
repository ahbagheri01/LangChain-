from dotenv import load_dotenv
import os
load_dotenv()
print(os.environ)


# from langchain.llms import OpenAI
#
# llm = OpenAI(
#     model_name=os.environ.get("LLM_MODEL_NAME"),  # e.g., "gpt-3.5-turbo" or anything you want
#     openai_api_key=os.environ.get("LLM_API_KEY"),  # dummy if not checked on server
#     openai_api_base=os.environ.get("LLM_API_BASE"),  # your local server address
#     temperature=0.7
# )
#
# response = llm
# print(response)

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model(
    model=os.environ.get("LLM_MODEL_NAME"),  # Can be any name; just consistent with your server
    model_provider=os.environ.get("LLM_MODEL_PROVIDER"),
    base_url=os.environ.get("LLM_API_BASE"),  # your local API endpoint
    openai_api_key=os.environ.get("LLM_API_KEY"),          # dummy if your server doesn’t check it
    temperature=0.3,
)

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]
# print(model.invoke(messages))
#
# print(model.invoke("Hello"))
#
# print(model.invoke([{"role": "user", "content": "Hello"}]))
#
# print(model.invoke([HumanMessage("Hello")]))
#
# for token in model.stream(messages):
#     print(token.content, end="|")
#
#
from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)