from .embeddings import model, collection

def retrieve_context(query, top_k=5):
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results['documents'][0] if results['documents'] else []