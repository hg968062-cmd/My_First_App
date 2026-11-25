
import os
import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings("ignore")

# ==============================
# IMPORT LIBRARIES
# ==============================

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================
# 1. STREAMLIT PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Statistical Techniques AI Tutor",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Statistical Techniques AI Tutor")

st.markdown(
    """
    Welcome to your **AI-powered Statistical Techniques Tutor**.

    Upload or place your statistics PDF in the project folder
    and ask questions based on its content.
    """
)

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("NVIDIA_API_KEY")


# ==============================
# 2. CHECK NVIDIA API KEY
# ==============================

if not api_key:
    st.error(
        "NVIDIA API KEY not found. "
        "Please add NVIDIA_API_KEY to your .env file."
    )
    st.stop()


# ==============================
# 3. AUTOMATIC PDF FINDING
# ==============================

def find_pdf():

    possible_paths = [
        BASE_DIR / "document_loader" / "STATS.pdf",
        BASE_DIR / "STATS.pdf",
        BASE_DIR / "documents" / "STATS.pdf"
    ]

    # Check predefined locations
    for path in possible_paths:
        if path.is_file():
            return path

    # Search other project subfolders
    matches = list(BASE_DIR.rglob("STATS.pdf"))

    if matches:
        return matches[0]

    return None


pdf_path = find_pdf()

# Allow uploading a PDF if automatic detection fails
if pdf_path is None:

    st.warning("STATS.pdf was not found in your project folders.")

    uploaded_file = st.file_uploader(
        "Upload your Statistical Techniques PDF",
        type=["pdf"]
    )

    if uploaded_file is None:
        st.info("Please upload your PDF to continue.")
        st.stop()

    # Save uploaded PDF
    upload_dir = BASE_DIR / "document_loader"
    upload_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = upload_dir / "uploaded_document.pdf"

    pdf_path.write_bytes(uploaded_file.getvalue())

st.success(f"PDF found: {pdf_path.name}")


# ==============================
# 4. LOAD PDF AND CREATE VECTORSTORE
# ==============================

@st.cache_resource(show_spinner="Loading PDF and creating vector database...")

def create_vectorstore(file_path, file_modified):

    # Load PDF
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    if not docs:
        raise ValueError("PDF contains no pages.")

    # Document chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text extracted from PDF.")

    # Initialize embeddings
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Calculate vector dimension
    vector_size = len(
        embeddings.embed_query("Test embedding")
    )

    # Create in-memory Qdrant database
    client = QdrantClient(location=":memory:")

    collection_name = "rag_collection"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # Create vectorstore
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Add documents
    vectorstore.add_documents(chunks)

    return vectorstore, len(docs), len(chunks)


# ==============================
# 5. INITIALIZE VECTOR DATABASE
# ==============================

try:

    vectorstore, total_pages, total_chunks = create_vectorstore(
        str(pdf_path),
        pdf_path.stat().st_mtime
    )

    st.success("PDF indexed successfully!")

    col1, col2 = st.columns(2)

    col1.metric("Total PDF Pages", total_pages)
    col2.metric("Total Text Chunks", total_chunks)

except Exception as e:

    st.error(f"Document processing error: {e}")
    st.stop()


# ==============================
# 6. INITIALIZE NVIDIA LLM
# ==============================

@st.cache_resource

def initialize_llm(key):

    return ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=key,
        temperature=0.2,
        max_tokens=2500
    )


try:

    llm = initialize_llm(api_key)

except Exception as e:

    st.error(f"NVIDIA initialization error: {e}")
    st.stop()


# ==============================
# 7. RETRIEVER
# ==============================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==============================
# 8. RAG PROMPT TEMPLATE
# ==============================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are an expert tutor in Statistical Techniques.

        Answer the question using ONLY the provided context.

        Instructions:

        1. Explain answers in simple and clear language.
        2. Provide formulas wherever necessary.
        3. Use examples when the context supports them.
        4. Do not invent information.
        5. If the answer is unavailable in the context,
           say that you don't know based on the provided document.

        Context:
        {context}
        """
    ),

    ("human", "{question}")

])


# ==============================
# 9. CREATE RAG CHAIN
# ==============================

rag_chain = prompt | llm | StrOutputParser()


# ==============================
# 10. QUERY RAG FUNCTION
# ==============================

def query_rag(question):

    if not question or not question.strip():
        return None, []

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "No relevant information found.", []

    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Generate answer
    response = rag_chain.invoke({

        "context": context,

        "question": question

    })

    return response, retrieved_docs


# ==============================
# 11. STREAMLIT USER INTERFACE
# ==============================

st.divider()

st.subheader("💬 Ask Your Question")

question = st.text_input(
    "Enter your Statistical Techniques question:",
    placeholder="Example: What is the principle of least squares?"
)

if st.button("🚀 Get Answer", use_container_width=True):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching PDF and generating your answer..."
        ):

            try:

                answer, retrieved_docs = query_rag(question)

                st.subheader("📝 AI Generated Answer")

                st.markdown(answer)

                # Display source documents
                with st.expander("📄 View Retrieved PDF Sources"):

                    for i, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        st.markdown(
                            f"**Source {i} | Page {page}**"
                        )

                        st.write(doc.page_content)

                        st.divider()

            except Exception as e:

                st.error(f"Error generating answer: {e}")


# ==============================
# 12. FOOTER
# ==============================

st.divider()

st.caption(
    "Developed using Python, Streamlit, LangChain, "
    "Qdrant, FastEmbed and NVIDIA Nemotron."
)
