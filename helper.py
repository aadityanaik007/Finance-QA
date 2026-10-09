from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

CHROMA_DIR = './chroma_db'
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def txt_splittor():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 100,
        separators=["\n\n","\n", " "],
        add_start_index=True
    )

def vector_db():
    return Chroma(
        collection_name="financial_docs",
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_DIR,
    )