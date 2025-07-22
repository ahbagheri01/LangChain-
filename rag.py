from dotenv import load_dotenv
load_dotenv()
from llm_api import get_model
from prompt_api import get_sample
from work_flow_api import *
from emb_api import get_emb_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.document_loaders import DirectoryLoader
print("testing emb")
emb_model = get_emb_model()
print("loading dir")
loader = DirectoryLoader("../../../codes/RSO/crawler/sc/webcrawler/html_storage/rso-co.ir/", glob="**/*.md",show_progress=True)
docs = loader.load()[:2]
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True,)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

vector_store = InMemoryVectorStore(emb_model)
llm_model = get_model()



batch_size = 16
results = []

for i in range(0, len(all_splits), batch_size):
    batch = all_splits[i:i + batch_size]
    result = vector_store.add_documents(documents=batch)
    results += result
print(len(results))
print(len(vector_store.store.keys()))
from langchain_core.prompts import ChatPromptTemplate
import os
system_template = ("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}""")
prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{question}")]
    )


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = llm_model.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

from IPython.display import Image, display

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

# Generate and save the PNG
png_bytes = graph.get_graph().draw_mermaid_png(
    output_file_path="workflow.png",         # ← specify your filename here
    draw_method=MermaidDrawMethod.API,      # or PYPPETEER if you prefer
)

# response = graph.invoke({"question": "محصول ESB"})
# print(response["answer"])