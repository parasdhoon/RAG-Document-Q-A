import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever import create_retriever
from llm import create_rag_chain, generate_response

if __name__=="__main__":
    try:
        user_question = "What is Generative AI"
        chat_history = []
        retriever = create_retriever()
        rag_chain = create_rag_chain(retriever)
        answer = generate_response(rag_chain, user_question, chat_history)
        
        print(f"Answer: {answer}")
        print(f"chat_history: {chat_history}")
    except Exception as e:
        print(f"Error generating the response: {e}")