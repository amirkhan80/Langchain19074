import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ------------------------------
# Streamlit UI setup
# ------------------------------
st.set_page_config(page_title="Offline PDF QA", layout="wide")
st.title("📄 Offline PDF/TXT QA (No Transformers, Fully Offline)")
st.write("Upload a PDF or TXT file and ask questions. Works completely offline, point-wise answers.")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    text = ""

    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            st.error("No readable text found in the PDF. Make sure it’s text-based (not scanned).")
            st.stop()
    else:
        text = uploaded_file.read().decode("utf-8")
        if not text.strip():
            st.error("The uploaded TXT file is empty.")
            st.stop()

    st.success("✅ Document uploaded successfully!")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    doc_objects = [Document(page_content=chunk) for chunk in chunks]

    if len(doc_objects) == 0:
        st.error("No text found to process.")
        st.stop()

    # ------------------------------
    # TF-IDF embeddings (offline)
    # ------------------------------
    corpus = [doc.page_content for doc in doc_objects]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ------------------------------
    # Ask questions
    # ------------------------------
    query = st.text_input("Ask a question about your document:")

    if query:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_idx = similarities.argsort()[::-1][:3]  # top 3 chunks

        answers = []
        for idx in top_idx:
            snippet = doc_objects[idx].page_content.strip()
            snippet = re.sub(r"^.*?(Chapter|Learning Objectives)", r"\1", snippet, flags=re.DOTALL)

            points = re.findall(r"(?:\d+\.|\-)\s.*?(?=(?:\d+\.|\-|$))", snippet, flags=re.DOTALL)
            if points:
                for p in points[:10]:  # limit to top 10 points
                    clean_p = p.strip().replace("\n", " ")
                    answers.append(f"- {clean_p}")
            else:
                clean_snip = snippet.replace("\n", " ")
                answers.append(f"- {clean_snip[:250]}...")

        final_answer = "\n".join(answers)
        st.markdown("### 📌 Answer (Point-wise)")
        st.write(final_answer)

        with st.expander("🔍 Full relevant snippets"):
            for i in top_idx:
                snippet = doc_objects[i].page_content[:800].replace("\n", " ")
                st.markdown(f"**Snippet {i+1}:** {snippet}...")
