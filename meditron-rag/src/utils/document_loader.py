"""
Document loader module for processing PDF files.
"""
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load_documents(docs_dir: str) -> List:
    """Load and process PDF documents.
    
    Args:
        docs_dir: Directory containing PDF files
        
    Returns:
        List of processed document chunks
    """
    try:
        # Get all PDF files
        pdf_files = list(Path(docs_dir).glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {docs_dir}")
            
        logger.info(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            logger.info(f"  - {pdf.name}")
        
        # Initialize text splitter with settings optimized for sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased from 800 for larger chunks
            chunk_overlap=300,  # Increased from 200 for better context preservation
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n\n",  # Large section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                "? ",      # Questions
                "! ",      # Exclamations
                " ",       # Words
                ""        # Characters
            ]
        )
        
        # Process each PDF
        all_chunks = []
        seen_contents = set()  # Track unique contents
        for pdf_path in pdf_files:
            try:
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                # Split into chunks
                chunks = text_splitter.split_documents(pages)
                
                # Add source to metadata and deduplicate
                for chunk in chunks:
                    chunk.metadata["source"] = pdf_path.name
                    # Only add if content is unique
                    if chunk.page_content not in seen_contents:
                        seen_contents.add(chunk.page_content)
                        all_chunks.append(chunk)
                
                logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue
                
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise 