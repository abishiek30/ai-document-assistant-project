import hashlib
import os

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()
st.set_page_config(page_title="AI Document Assistant")

st.title("📚 AI Document Assistant")
st.write("Ask questions about your PDF using Gemini AI and RAG.")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        api_key = None

if not api_key:
    st.error(
        "Gemini API key not found. Set GEMINI_API_KEY in your .env file "
        "or in your Streamlit app secrets."
    )
    st.stop()

client = genai.Client(
    api_key=api_key,
    http_options=genai.types.HttpOptions(timeout=60000),
)


@st.cache_resource(max_entries=1)
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to start asking questions about it.")
    st.stop()

file_bytes = uploaded_file.getvalue()
document_id = hashlib.sha256(file_bytes).hexdigest()

if (
    st.session_state.get("document_id") != document_id
    or "document_vector_store" not in st.session_state
):
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            extracted_text = "\n".join(page.get_text("text") for page in pdf)
    except Exception:
        st.error("Could not read this PDF. Make sure it is valid and not password-protected.")
        st.stop()

    if not extracted_text.strip():
        st.error("No selectable text found in this PDF. Scanned PDFs need OCR first.")
        st.stop()

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        st.error("PDF processing packages are missing. Install them with: python -m pip install -r requirements.txt")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(extracted_text)
    if not chunks:
        st.error("No text could be extracted from this PDF.")
        st.stop()

    try:
        with st.spinner("Creating the document search index..."):
            vector_store = FAISS.from_texts(chunks, load_embeddings())
    except Exception:
        st.error(
            "Could not create the document search index. Check your internet connection "
            "and try uploading the PDF again."
        )
        st.stop()

    st.session_state.document_id = document_id
    st.session_state.document_vector_store = vector_store
    st.session_state.document_chunk_count = len(chunks)
    st.session_state.pop("answer", None)
    st.session_state.pop("answer_context", None)

st.success("PDF processed successfully!")
st.write("Total chunks:", st.session_state.document_chunk_count)

with st.form("document_question", clear_on_submit=False):
    question = st.text_input("Ask a question about your document")
    submitted = st.form_submit_button("Ask Gemini", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question about the PDF first.")
    else:
        try:
            results = st.session_state.document_vector_store.similarity_search(
                question.strip(), k=3
            )
            context_parts = [result.page_content for result in results]
            context = "\n\n".join(context_parts)
            prompt = f"""You are an AI document assistant.

Answer the user's question using only the information provided in the document context.
If the answer is not available in the context, say that the information is not found in the document.

Document Context:
{context}

User Question:
{question.strip()}

Answer:
"""
            with st.spinner("Gemini is generating your answer..."):
                response = client.interactions.create(
                    model="gemini-3.5-flash-lite",
                    input=prompt,
                )

            answer = response.output_text
            if not answer:
                st.warning("Gemini returned an empty response. Please try again.")
            else:
                st.session_state.answer = answer
                st.session_state.answer_context = context_parts
                st.session_state.answer_document_id = document_id
        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code:
                st.error(
                    f"Gemini request failed (HTTP {status_code}). Check your API key, "
                    "quota, and Gemini API access, then try again."
                )
            else:
                st.error("Gemini could not answer right now. Check your API key and try again.")
        except Exception:
            st.error("Something went wrong while preparing the answer. Please try again.")

if (
    st.session_state.get("answer")
    and st.session_state.get("answer_document_id") == document_id
):
    st.subheader("🤖 AI Generated Answer")
    st.write(st.session_state.answer)

    with st.expander("View retrieved document context"):
        for index, passage in enumerate(st.session_state.answer_context, start=1):
            st.write(f"**Source {index}**")
            st.write(passage)
