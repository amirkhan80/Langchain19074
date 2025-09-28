import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Offline PDF/TXT QA", layout="wide")
st.title("📄 Offline PDF/TXT QA (Point-wise Answers, No Transformers)")
st.write("Upload a PDF or TXT file and ask questions. Outputs **point-wise answers** without using torch/transformers.")

# ------------------------------
# File upload
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
            st.error("No readable text found in PDF. Make sure it’s text-based.")
            st.stop()
    else:  # TXT file
        text = uploaded_file.read().decode("utf-8")
        if not text.strip():
            st.error("Uploaded TXT is empty.")
            st.stop()

    st.success("✅ Document uploaded successfully!")

    # ------------------------------
    # Split text into chunks
    # ------------------------------
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    if len(chunks) == 0:
        st.error("No text found to process.")
        st.stop()

    # ------------------------------
    # Build TF-IDF vectorizer
    # ------------------------------
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)

    # ------------------------------
    # Ask questions
    # ------------------------------
    query = st.text_input("Ask a question about your document:")

    if query:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx = similarities.argsort()[-3:][::-1]  # top 3 chunks

        answers = []
        for idx in top_idx:
            snippet = chunks[idx].strip()
            snippet = re.sub(r"^.*?(Chapter|Learning Objectives)", r"\1", snippet, flags=re.DOTALL)
            points = re.findall(r"(?:\d+\.|\-)\s.*?(?=(?:\d+\.|\-|$))", snippet, flags=re.DOTALL)
            if points:
                for p in points[:10]:
                    clean_p = p.strip().replace("\n", " ")
                    answers.append(f"- {clean_p}")
            else:
                clean_snip = snippet.replace("\n", " ")
                answers.append(f"- {clean_snip[:250]}...")

        final_answer = "\n".join(answers)

        st.markdown("### 📌 Answer (Point-wise)")
        st.write(final_answer)

        # Show full snippets
        with st.expander("🔍 Full relevant snippets"):
            for i in top_idx:
                snippet = chunks[i].replace("\n", " ")
                st.markdown(f"**Snippet {i+1}:** {snippet[:800]}...")
