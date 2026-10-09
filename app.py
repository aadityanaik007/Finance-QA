import os
import streamlit as st
from ingest import ingest_pdf
from graph import graph

UPLOAD_DIR = "pdf_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Financial Document Q&A")
st.title("Financial Document Q&A")

# Ingest a PDF
uploaded = st.file_uploader("Upload a financial PDF", type="pdf")
if uploaded is not None and st.button("Ingest document"):
    path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    with st.spinner("Ingesting... this can take a few minutes"):
        n = ingest_pdf(path, uploaded.name)
    st.success(f"Stored {n} chunks from {uploaded.name}")

st.divider()

# Ask a question
question = st.text_input("Ask a question about your documents")
if question:
    with st.spinner("Searching and answering..."):
        result = graph.invoke({"question": question})
    st.markdown(result["answer"])