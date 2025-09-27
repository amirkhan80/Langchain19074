import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ------------------------------
# Streamlit UI setup
# ------------------------------
st.set_page_config(page_title="Offline Transformer QA", layout="wide")
st.title("📄 Offline Transformer QA")
st.write("Upload a PDF or TXT file and ask questions about its content. No API key needed!")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    text = ""

    # ---------- PDF Handling ----------
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if not text.strip():
            st.error("No readable text found in the PDF. Make sure it’s text-based (not scanned).")
            st.stop()

    # ---------- TXT Handling ----------
    else:
        text = uploaded_file.read().decode("utf-8")
        if not text.strip():
            st.error("The uploaded TXT file is empty.")
            st.stop()

    st.success("Document uploaded successfully!")

    # ------------------------------
    # Split text into chunks
    # ------------------------------
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    doc_objects = [Document(page_content=chunk) for chunk in chunks]

    if len(doc_objects) == 0:
        st.error("No text found to process.")
        st.stop()

    # ------------------------------
    # Generate embeddings and FAISS vector store
    # ------------------------------
    with st.spinner("Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(doc_objects, embeddings)

    # ------------------------------
    # Initialize local HuggingFace QA model (CPU)
    # ------------------------------
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        device=-1  # CPU only
    )

    # ------------------------------
    # Ask questions
    # ------------------------------
    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Searching for answer..."):
            try:
                # Retrieve top 3 relevant chunks from FAISS
                docs_with_scores = vectorstore.similarity_search(query, k=3)

                # Run QA only on these top chunks
                answers = []
                for doc in docs_with_scores:
                    result = qa_pipeline(question=query, context=doc.page_content)
                    if result['answer'].strip():
                        answers.append(result['answer'].strip())

                # Combine answers
                if not answers:
                    final_answer = "No relevant information found in the document."
                else:
                    final_answer = " ".join(answers)

                st.markdown(f"**Answer:** {final_answer}")

                # Optional: show snippet sources
                with st.expander("Show relevant snippets"):
                    for i, doc in enumerate(docs_with_scores):
                        snippet = doc.page_content[:300].replace("\n", " ")
                        st.markdown(f"**Snippet {i+1}:** {snippet}...")

            except Exception as e:
                st.error(f"Error generating answer: {e}")