from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("NVIDIA_EMBEDDINGS_API_KEY")

def embed_documents(final_docs):
    if not final_docs:
        return
    
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_dir = os.path.join(base_dir, "data", "chromadb")
        
        if not os.path.exists(chroma_dir):
            os.makedirs(chroma_dir)
        
        embedding_model = NVIDIAEmbeddings(
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key="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"
        )
        
        vectorstoredb = Chroma(
            collection_name="chroma_indexes",
            embedding_function=embedding_model,
            persist_directory=chroma_dir
        )
        
        vectorstoredb.add_documents(final_docs)
        
        print("Documents Embeded Successfully!")
    except Exception as e:
        raise Exception(f"Error Embedding the documents: {e}")