"""
Vector store module for document storage and retrieval.
"""
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import chromadb
import os
import shutil

from config.settings import VECTOR_STORE_PATH, MAX_DOCUMENTS

logger = logging.getLogger(__name__)

def initialize_vectorstore(
    embeddings: Embeddings,
    documents: List[Document],
    persist_directory: Optional[str] = None
) -> Chroma:
    """Initialize vector store with documents.
    
    Args:
        embeddings: Embeddings model
        documents: List of documents to add
        persist_directory: Directory to persist vector store
        
    Returns:
        Initialized vector store
    """
    try:
        # Create fresh persist directory
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory)
        
        # Deduplicate documents by content hash
        seen_hashes = set()
        unique_docs = []
        for doc in documents:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
        
        logger.info(f"Deduplication reduced documents from {len(documents)} to {len(unique_docs)}")
        
        # Initialize vector store with unique documents
        vector_store = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        
        logger.info(f"Initialized vector store with {len(unique_docs)} documents at {persist_directory}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = MAX_DOCUMENTS
) -> List[Document]:
    """Perform similarity search in vector store.
    
    Args:
        vector_store: Chroma vector store
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    try:
        docs = vector_store.similarity_search(
            query,
            k=k,
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,  # Fetch more candidates for better results
                "lambda_mult": 0.5  # Control diversity of results
            }
        )
        logger.info(f"Retrieved {len(docs)} documents for query")
        return docs
        
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return [] 