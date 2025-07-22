from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm_api import get_model
# Define a new graph

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_sample_w():
    workflow = StateGraph(state_schema=MessagesState)
    model = get_model()

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}



# Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

# Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def get_sample_w2():


    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You talk like a pirate. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    workflow = StateGraph(state_schema=MessagesState)
    model = get_model()

    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": response}



# Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

# Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    name: str
def get_sample_w3():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your name is {name} talk like a person from {language} Speaking in {language}. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    workflow = StateGraph(state_schema=State)
    model = get_model()

    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

