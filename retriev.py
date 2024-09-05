from langchain.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.llms import Ollama  # Replace with the LLM you are using
from langchain.embeddings import OpenAIEmbeddings  # Replace with your embedding model

# 1. Define the embedding model (the same one you used for embedding documents)
embeddings_model = OpenAIEmbeddings()  # Replace with Ollama's embedding model if you're using that

# 2. Connect to the Milvus database
vector_store = Milvus(
    collection_name="rfp_docs2", 
    embedding_function=embeddings_model, 
    connection_args={"host": "localhost", "port": "19530"}
)

# 3. Set up the RetrievalQA chain
llm = Ollama()  # Replace with the LLM you're using
retriever = vector_store.as_retriever()

qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# 4. Ask a question to retrieve information
query = "What is the timeline for the RFP?"
result = qa_chain.run(query)

print(result)
