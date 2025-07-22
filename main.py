from dotenv import load_dotenv
from llm_api import get_model
from prompt_api import get_sample
from work_flow_api import get_sample_w
load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

app = get_sample_w()
config = {"configurable": {"thread_id": "abc123"}}
query = "Hi! I'm Bob."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state



query = "What's my name?"

config = {"configurable": {"thread_id": "abc123"}}
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

print("DONE")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)