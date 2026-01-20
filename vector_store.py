from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_documents, split_documents

VECTOR_DB_PATH = "vector_store/faiss_index"

def build_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs("vector_store", exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    print("Vector store created and saved successfully.")

if __name__ == "__main__":
    build_vector_store()
