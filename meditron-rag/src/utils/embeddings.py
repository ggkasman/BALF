"""
Embeddings module for document vectorization.
"""
import logging
import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def create_embeddings() -> HuggingFaceEmbeddings:
    """Create embeddings model for document vectorization.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        # Force CPU usage to avoid memory issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Set torch to use CPU and limit threads
        torch.set_num_threads(4)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/S-PubMedBert-MS-MARCO",
            model_kwargs={'device': 'cpu'},
            cache_folder="./cache/embeddings"
        )
        
        logger.info("Created embeddings model using PubMedBERT")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings model: {str(e)}")
        raise 