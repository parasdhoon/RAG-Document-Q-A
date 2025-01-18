from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        raw_dir = os.path.join(base_dir, "data", "raw")
        
        if not os.path.exists(raw_dir):
            raise Exception("No file Uploaded")
        
        loader = PyPDFDirectoryLoader(path=raw_dir)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(docs)
        
        return final_docs
    except Exception as e:
        print("Error loading the documents: {e}")