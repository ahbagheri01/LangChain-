from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm_api import get_model
# Define a new graph




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