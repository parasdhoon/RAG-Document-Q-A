import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from loaders import load_documents
from embed import embed_documents
from retriever import create_retriever

def main():
    retriever = None
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    st.set_page_config(
        page_title="Rag Document Q&A",
        page_icon="📚"
    )
    st.title("RAG Document Q&A")
    
    uploaded_files = st.file_uploader("Uploaqd your pdf documents here", accept_multiple_files=True)
    
    if st.button("Read Documents"):
        with st.spinner("Reading Documents📄"):
            for uploaded_file in uploaded_files:
                raw_file_path = os.path.join(raw_dir, uploaded_file.name)
                
                with open(raw_file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
            
            final_docs = load_documents()
            embed_documents(final_docs)
            
            retriever = create_retriever()
            st.success("Ready for your questions")
    
    user_question = st.text_input("Enter the question here:")
    
    if st.button("Generate Answer"):
        with st.spinner("Generating Answer"):
            
            if retriever is None:
                st.warning("No documents uploaded to read!")
            
            if not user_question:
                st.warning("Enter Question to ask.")