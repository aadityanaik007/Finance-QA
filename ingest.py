from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from helper import txt_splittor,vector_db

load_dotenv()

def ingest_pdf(path:str,doc_name:str)->int:
    # Load Document
    loader = PyPDFLoader(
    file_path = f"{path}")
    pages = loader.load()
    vector_db_obj = vector_db()

    # Split docs into chunks
    chunks = txt_splittor().split_documents(pages)
    
    for chunk in chunks:
        chunk.metadata['source_doc'] = doc_name

    # Add it to Chroma DB
    vector_db_obj.add_documents(chunks)

    return len(chunks)

print(ingest_pdf("pdf_files/10-k-Annual-report.pdf","10-k-Annual-report.pdf"))