
import os
import streamlit as st
import fitz

from dotenv import load_dotenv
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="AI Document Assistant")

st.title("📚 AI Document Assistant")
st.write("Ask questions about your PDF using Gemini AI and RAG.")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

client = genai.Client(
    api_key=api_key,
    http_options=genai.types.HttpOptions(timeout=30000),
)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    extracted_text = ""

    for page in pdf:
        extracted_text += page.get_text("text") + "\n"

    if not extracted_text.strip():
        st.error("No selectable text found in this PDF.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(extracted_text)

    st.success("PDF processed successfully!")
    st.write("Total chunks:", len(chunks))

    embeddings = load_embeddings()

    with st.spinner("Creating FAISS database..."):
        vector_store = FAISS.from_texts(
            chunks,
            embeddings
        )

    st.success("FAISS database is ready!")

    question = st.text_input(
        "Ask a question about your document"
    )

    if question:

        with st.spinner("Gemini is generating your answer..."):

            results = vector_store.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                result.page_content for result in results
            )

            prompt = f"""
You are an AI document assistant.

Answer the user's question using only the
information provided in the document context.

If the answer is not available in the context,
say that the information is not found in the document.

Document Context:
{context}

User Question:
{question}

Answer:
"""

            try:
                response = client.interactions.create(
                    model="gemini-3.5-flash-lite",
                    input=prompt
                )

                st.subheader("🤖 AI Generated Answer")
                st.write(response.output_text)

                with st.expander("View Retrieved Document Context"):
                    for i, result in enumerate(results):
                        st.write(f"**Source {i + 1}**")
                        st.write(result.page_content)

            except Exception as e:
                st.error(f"Gemini API error: {e}")

