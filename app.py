import streamlit as st
from rag_chain import load_rag_chain

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Healthcare GenAI Research Assistant",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Session State Initialization
# -------------------------------
if "query" not in st.session_state:
    st.session_state.query = ""

# -------------------------------
# Header
# -------------------------------
st.title("🩺 Healthcare GenAI Research Assistant")
st.caption("Powered by WHO & CDC medical documents")

# -------------------------------
# Medical Disclaimer
# -------------------------------
st.warning(
    "⚠️ **Disclaimer:** This tool is for educational and research purposes only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)

# -------------------------------
# Load RAG Chain (Cached)
# -------------------------------
@st.cache_resource
def get_rag():
    return load_rag_chain()

rag = get_rag()

# -------------------------------
# Example Questions
# -------------------------------
st.markdown("### Example Questions")
example_questions = [
    "What are the symptoms of Type 2 Diabetes?",
    "How is diabetes diagnosed?",
    "Can diabetes cause kidney problems?",
    "What lifestyle changes help manage diabetes?"
]

cols = st.columns(2)
for i, q in enumerate(example_questions):
    if cols[i % 2].button(q):
        st.session_state.query = q

# -------------------------------
# Question Input (BOUND TO STATE)
# -------------------------------
st.markdown("### Ask a healthcare-related question")
st.text_input(
    "",
    key="query",
    placeholder="Type your medical question here..."
)

# -------------------------------
# Get Answer
# -------------------------------
if st.button("Get Answer"):
    if not st.session_state.query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching trusted medical sources..."):
            response = rag.invoke(st.session_state.query)

        # -------------------------------
        # Answer Section
        # -------------------------------
        st.markdown("## Answer")
        st.write(response.content)

        # -------------------------------
        # Safety / Confidence Note
        # -------------------------------
        st.info(
            "This response is generated using authoritative public health documents "
            "and is intended strictly for educational and research purposes."
        )

        # -------------------------------
        # Source Transparency
        # -------------------------------
        with st.expander("Sources used (WHO / CDC)"):
            st.write(
                "- World Health Organization (WHO)\n"
                "- Centers for Disease Control and Prevention (CDC)"
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <div style="margin-top:40px; padding-top:10px; border-top:1px solid #2c2c2c;
                text-align:center; font-size:0.85rem; color:#9aa0a6;">
        Built with RAG, LangChain, FAISS & Streamlit •
        <a href="https://github.com/sujal-kadam/Healthcare-AI-Assistant"
           target="_blank" style="color:#8ab4f8; text-decoration:none;">
           GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

