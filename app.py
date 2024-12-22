### RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="Gemma2-9b-It", max_tokens=5000)


## Set Up StreamLit
st.title("Conversational RAG With PDF uploads and Chat History")
st.write("Upload PDF's and chat with their content")

## Chat Interface
session_id = st.text_input("Session ID", value="default_session")
## Statefully manage the chat history

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

## Process Uploaded PDF's
if uploaded_files:
    ## Data Ingestion
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    
    ## Splitting and Creating Document Embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma(
        collection_name="test_collections",
        embedding_function=embeddings,
        persist_directory="./chroma"
    )
    vectorstore.add_documents(splits)
    retriever = vectorstore.as_retriever()

    ## Chat History prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might refer context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed otherwise return it as it is"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    ## Question Answer Prompt
    system_prompt = (
        "You are a powerful assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer"
        "the question in full detail. If you don't know the answer, say that you"
        "don't know, but do not ask the user for more information. Provide a detailed"
        "explanation and a comprehensive response unless explicitly stated otherwise."
        "\n\n"
        "{context}"
    )


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    ## Rag Chain with Message History
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    user_input = st.text_input("Your Question?:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config = {
                "configurable": {"session_id": session_id}
            },
        )
        
        st.write(st.session_state.store)
        st.success(f"Assistant: {response['answer']}")
        st.write("Chat History: ", session_history.messages)
else:
    st.warning("Please Upload the files")