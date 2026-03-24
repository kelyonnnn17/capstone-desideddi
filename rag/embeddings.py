from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()
collection = client.get_or_create_collection("ddi_knowledge")

def add_documents(docs):
    embeddings = model.encode(docs)

    ids = [str(i) for i in range(len(docs))]

    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        ids=ids
    )