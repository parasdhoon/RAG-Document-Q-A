import streamlit as st
import os
import glob
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from loaders import load_documents
from embed import embed_documents
from retriever import create_retriever
from llm import create_rag_chain, generate_response

def clear_dir(dir):
    files = glob.glob(os.path.join(dir, '*'))

    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Could not delete {file}. Reason: {e}")

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    chromadb_dir = os.path.join(base_dir, 'data', 'faiss_index')
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    
    st.set_page_config(
        page_title="Rag Document Q&A",
        page_icon="📚"
    )
    st.title("RAG Document Q&A")
    
    uploaded_files = st.file_uploader("Upload your pdf documents here", accept_multiple_files=True)
    
    if st.button("Read Documents"):
        with st.spinner("Reading Documents📄"):
            clear_dir(raw_dir)
            clear_dir(chromadb_dir)
            for uploaded_file in uploaded_files:
                raw_file_path = os.path.join(raw_dir, uploaded_file.name)
                
                with open(raw_file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
            
            final_docs = load_documents()
            embed_documents(final_docs)
            
            st.session_state.retriever = create_retriever()
            st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
            st.success("Ready for your questions")
    
    user_question = st.text_input("Enter the question here:")
    
    if st.button("Generate Answer"):
        with st.spinner("Generating Answer"):
            
            if st.session_state.rag_chain is None:
                st.warning("No documents uploaded to read!")
            elif not user_question:
                st.warning("Enter Question to ask.")
            else:
                answer = generate_response(st.session_state.rag_chain, user_question, st.session_state.chat_history)
                
                if not answer:
                    st.warning("Sorry, unable to generate the response at the moment.")
                else:
                    st.success(f"Assistant:\n{answer}")