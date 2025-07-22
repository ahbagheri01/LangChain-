from dotenv import load_dotenv
from llm_api import get_model
from prompt_api import get_sample
from work_flow_api import *
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

app2 = get_sample_w2()
config = {"configurable": {"thread_id": "abc3443"}}
query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app2.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app2.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()



app3 = get_sample_w3()
config = {"configurable": {"thread_id": "abc3443"}}
query = "Hi! I'm Jim."
language = "Persia"
input_messages = [HumanMessage(query)]
output = app3.invoke({"messages": input_messages, "language": language, "name":"AMIR"}, config)
output["messages"][-1].pretty_print()

query = "What is your name?"

input_messages = [HumanMessage(query)]
output = app3.invoke({"messages": input_messages, "language": language, "name":"AMIR"}, config)
output["messages"][-1].pretty_print()