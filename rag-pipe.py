import os
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

# Connect to Milvus
def connect_to_milvus(host='host.docker.internal', port='19530'):
    connections.connect(host=host, port=port)
    print("Connected to Milvus")

# Retrieve documents from Milvus
def retrieve_from_milvus(collection_name, query, top_k=3):
    collection = Collection(collection_name)
    collection.load()

    # Generate embedding for the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0].tolist()

    # Search in Milvus
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    # Extract and return the text of the retrieved documents
    retrieved_docs = [hit.entity.get('text') for hit in results[0]]
    return retrieved_docs

# Format context for Ollama
def format_context(retrieved_docs):
    return "\n\n".join(retrieved_docs)

# Make a call to Ollama
def query_ollama(prompt, context, model="llama2"):
    url = "http://host.docker.internal:11434/api/generate"
    data = {
        "model": model,
        "prompt": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Main RAG pipeline function
def rag_pipeline(query, collection_name="document_collection"):
    # Connect to Milvus
    connect_to_milvus()

    # Retrieve relevant documents
    retrieved_docs = retrieve_from_milvus(collection_name, query)

    # Format context
    context = format_context(retrieved_docs)

    # Query Ollama
    answer = query_ollama(query, context)

    return answer

# Example usage
if __name__ == "__main__":
    user_query = "What is the main topic of the documents?"
    response = rag_pipeline(user_query)
    print(f"Query: {user_query}")
    print(f"Response: {response}")