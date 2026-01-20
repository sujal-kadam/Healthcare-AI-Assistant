from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

VECTOR_DB_PATH = "vector_store/faiss_index"


def load_rag_chain():
    # Load embeddings
    embeddings = OpenAIEmbeddings()

    # Load vector store
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt with guardrails
    prompt = PromptTemplate.from_template(
        """
You are a healthcare research assistant.
Answer the question ONLY using the context below.
If the answer is not found in the context, say:
"I don't have enough information from the provided medical sources."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # RAG chain (modern way)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


if __name__ == "__main__":
    rag = load_rag_chain()

    while True:
        query = input("\nAsk a medical question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        response = rag.invoke(query)
        print("\nAnswer:\n", response.content)
