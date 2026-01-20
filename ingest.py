from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_DIR = "diabetes-medical_docs"

def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} pages")
    print(f"Created {len(chunks)} chunks")
