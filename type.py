# Check the schema of an existing collection
from pymilvus import Collection

# Connect to Milvus
connections.connect("default", host="host.docker.internal", port="19530")

# Get the collection and print the schema
collection = Collection("rfp_docs")
print(collection.schema)
