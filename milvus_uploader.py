import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import numpy as np
from docx import Document
import re

# Connect to Milvus
def connect_to_milvus(host='localhost', port='19530'):
    connections.connect(host=host, port=port)
    print("Connected to Milvus")

# Create a collection in Milvus
def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Document collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create an IVF_FLAT index for fast retrieval
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection: {collection_name}")
    return collection

# Generate embeddings for documents
def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    return embeddings.tolist()

# Upload documents to Milvus
def upload_to_milvus(collection, documents, embeddings):
    entities = [
        embeddings,  # List of embeddings
        documents,   # List of documents
    ]
    insert_result = collection.insert(entities)
    print(f"Inserted {insert_result.insert_count} entities")

# Read and process the .docx file
def process_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # Only include non-empty paragraphs
            full_text.append(para.text.strip())
    return full_text

# Read and process a plain text file
def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return [text]  # Return the full content as a list with a single element

# Split text into chunks
def split_into_chunks(text, max_chunk_size=500):
    chunks = []
    current_chunk = ""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Main function to process and upload documents
def process_and_upload_documents(file_path, collection_name="document_collection2"):
    # Connect to Milvus
    connect_to_milvus()
    
    # Create collection
    collection = create_collection(collection_name, dim=384)  # 384 is the dimension for 'all-MiniLM-L6-v2' model
    
    # Determine file type and process accordingly
    if file_path.endswith('.docx'):
        paragraphs = process_docx(file_path)
    elif file_path.endswith('.txt'):
        paragraphs = process_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .docx or .txt file.")
    
    # Split paragraphs into chunks
    chunks = []
    for paragraph in paragraphs:
        chunks.extend(split_into_chunks(paragraph))
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    # Upload to Milvus
    upload_to_milvus(collection, chunks, embeddings)
    
    print("Document processed and uploaded successfully")

# Example usage
if __name__ == "__main__":
    file_path = '/Users/zfeldstein/rfp-chatbot/rancher.txt'  # Change this to your file path (can be .txt or .docx)
    process_and_upload_documents(file_path)
