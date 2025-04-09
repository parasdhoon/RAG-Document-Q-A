from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RAG-QA-Project")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def create_rag_chain(retriever):
    llm = ChatGroq(
        model="Llama-3.3-70b-Specdec",
        temperature=0.7,
        max_tokens=None,
        max_retries=2,
        timeout=None
    )
    
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might refer context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed otherwise return it as it is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ('user', '{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    system_prompt = """
        You are a powerful assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer
        the question in full detail. If you don't know the answer, say that you
        don't know, but do not ask the user for more information. Provide a detailed
        explanation and a comprehensive response unless explicitly stated otherwise.
        \n\n
        {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder("chat_history"),
            ('user', '{input}')
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def generate_response(rag_chain, user_question, chat_history):
    try:
        llm = ChatGroq(
            model="Llama-3.3-70b-Specdec",
            temperature=0.7,
            max_tokens=None,
            max_retries=2,
            timeout=None
        )
        
        response = rag_chain.invoke({
            "input": user_question,
            "chat_history": chat_history
        })
        
        chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        chat_history.append({
            "role": "assistant",
            "content": response['answer']
        })
        
        return response['answer']
    except Exception as e:
        print(f"Error occurred while generating the response: {e}")
        return ""