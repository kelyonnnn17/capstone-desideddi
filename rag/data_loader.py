import json
from .embeddings import add_documents

def load_rag_data(file_path="data/rag_docs.json"):
    with open(file_path, "r") as f:
        docs = json.load(f)

    add_documents(docs)