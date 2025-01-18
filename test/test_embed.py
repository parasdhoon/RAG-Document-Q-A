import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from loaders import load_documents
from embed import embed_documents

if __name__ == "__main__":
    final_docs = load_documents()
    embed_documents(final_docs)