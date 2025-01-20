from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
#from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def create_retriever():
    try:
        #api_key = os.getenv("NVIDIA_EMBEDDINGS_API_KEY")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_dir = os.path.join(base_dir, "data", "chromadb")
        
        if not os.path.exists(chroma_dir):
            raise Exception("Indexes not found")
        
        # embedding_model = NVIDIAEmbeddings(
        #     model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        #     api_key=api_key
        # )
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        vectorstoredb = Chroma(
            collection_name="chroma_indexes",
            embedding_function=embedding_model,
            persist_directory=chroma_dir
        )
        
        retriever = vectorstoredb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        return retriever
    except Exception as e:
        print(f"Error Creating the retriever!")