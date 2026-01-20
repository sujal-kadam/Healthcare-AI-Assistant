# Healthcare GenAI Research Assistant (RAG-based)

An AI-powered healthcare research assistant that answers medical questions using authoritative public health documents from the World Health Organization (WHO) and the Centers for Disease Control and Prevention (CDC).
The system uses a Retrieval-Augmented Generation (RAG) approach to ensure responses are grounded, explainable, and free from hallucinations.

# Features

Question answering based on WHO & CDC medical documents
Retrieval-Augmented Generation (RAG) architecture
Semantic search using FAISS vector database
Hallucination control via strict prompt guardrails
Secure API key handling using .env
Streamlit-based professional healthcare UI
Example questions for quick demos
Source transparency and medical disclaimer

# Project Architecture
Document Ingestion Module – Loads and chunks WHO & CDC PDFs
Embedding & Vector Store Module – Generates embeddings and stores them in FAISS
RAG Pipeline – Retrieves relevant medical context and generates grounded answers\
Streamlit App – Interactive interface for asking healthcare questions

# Tech Stack
 
Python
Streamlit
LangChain (Runnable-based RAG)
FAISS
OpenAI API
PyPDF
python-dotenv

# Setup Instructions
```bash
git clone https://github.com/your-username/Healthcare-GenAI-Research-Assistant.git
cd Healthcare-GenAI-Research-Assistant
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

# Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key

# Ingest Documents & Build Vector Store
python ingest.py
python vector_store.py

# Run the Application
```bash
streamlit run streamlit_app.py
```

# Medical Disclaimer

This tool is intended only for educational and research purposes.
It is not a substitute for professional medical advice, diagnosis, or treatment.

## Author

**Sujal Kadam**  
B.Sc. IT | AI & Automation 
LinkedIn: https://www.linkedin.com/in/sujal-kadam-b824a7398/  
GitHub: https://github.com/sujal-kadam/Resume-Ai-Agent
