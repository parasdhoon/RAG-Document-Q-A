import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retriever import create_retriever

if __name__ == "__main__":
    retriever = create_retriever()
    
    print(retriever.invoke("What is Machine Learning?"))