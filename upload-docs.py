from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from langchain.embeddings import OpenAIEmbeddings  # Replace with your embedding model
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Define the schema for the collection, using 'vector' as the field name
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),  # Changed to 'vector'
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "Document Embeddings Collection")

# 3. Create the collection if it doesn't exist
collection = Collection(name="rfp_docs2", schema=schema)

# 4. Preprocess and generate embeddings for documents
documents = ["Your document text here", "Another document text here"]  # Replace with actual content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Initialize the embedding model here
embeddings_model = OpenAIEmbeddings()  # Replace with Ollama's embedding model if you're using that

# Initialize lists to store all embeddings and document chunks
all_embeddings = []
all_documents = []

# Split documents and generate embeddings
for doc in documents:
    chunks = text_splitter.split_text(doc)
    doc_embeddings = embeddings_model.embed_documents(chunks)  # Use the correct embedding model to generate embeddings
    all_embeddings.extend(doc_embeddings)
    all_documents.extend(chunks)

# 5. Insert embeddings and document chunks into Milvus
collection.insert([all_embeddings, all_documents])
collection.flush()

# 6. Create index on the 'vector' field
index_params = {
    "index_type": "IVF_FLAT",  # You can also use other index types like IVF_SQ8, etc.
    "metric_type": "L2",       # Use "L2" for Euclidean distance or "IP" for Inner Product
    "params": {"nlist": 100}
}

# Create the index on the 'vector' field
collection.create_index(field_name="vector", index_params=index_params)

print("Documents and embeddings uploaded and indexed in Milvus.")
