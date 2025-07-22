from dotenv import load_dotenv
import os
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
from langchain_postgres import PGVector

# See docker command above to launch a postgres instance with pgvector enabled.

print("testing emb")
emb_model = get_emb_model()
print("loading dir")
loader = DirectoryLoader("../../../codes/RSO/crawler/sc/webcrawler/html_storage/rso-co.ir/a/", glob="**/*.md",show_progress=True)
docs = loader.load()
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256, add_start_index=True,)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

connection = os.environ.get("PGVECTOR")  # Uses psycopg3!
collection_name = "RSO_DOC2"

vector_store = PGVector(
    embeddings=emb_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
llm_model = get_model()



batch_size = 16
results = []

for i in range(0, len(all_splits), batch_size):
    batch = all_splits[i:i + batch_size]
    result = vector_store.add_documents(documents=batch)
    results += result
