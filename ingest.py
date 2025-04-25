#!/usr/bin/env python3
"""
Document Ingestion Script for RAG Implementation

This script:
1. Loads documents from ./data/ directory
2. Processes them using semantic chunking
3. Embeds and stores them in Pinecone
"""

import os
import glob
import json
import logging
import re
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Configure OpenAI for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
DATA_DIR = Path("./data")
BASE_URL = "" 
MARKDOWN_DIR = DATA_DIR / "markdown"


def get_pinecone_client():
    """Create and return a Pinecone client."""

    # First initialize a controller client
    pc_controller = Pinecone(
        api_key=PINECONE_API_KEY
    )
        
    # Then create the index client with the specific host
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        host=PINECONE_HOST
    )
        
    # check if index exists
    pc_controller.describe_index(INDEX_NAME)

    logger.info(f"Successfully connected to Pinecone index: {INDEX_NAME}")
    return pc


def clear_pinecone_index(index_name: str = INDEX_NAME):
    """Delete all vectors in the Pinecone index."""
    logger.info(f"Clearing all vectors from Pinecone index: {index_name}")
    
    try:
        # Get Pinecone client
        pc = get_pinecone_client()
        if not pc:
            logger.error("Failed to initialize Pinecone client")
            return False
        
        # Get the index and delete all vectors
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        
        logger.info(f"Successfully cleared all vectors from index: {index_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {e}")
        return False


def get_source_url_from_path(file_path: str) -> str:
    """Convert a file path to a source URL."""
    
    # Get just the basename without the directory path
    if isinstance(file_path, Path):
        file_path = str(file_path)
        
    # Check if it's a absolute path or relative path
    if "data/markdown/" in file_path:
        # Locate the filename after data/markdown/
        match = re.search(r'data/markdown/([^/]+)$', file_path)
        if match:
            filename = match.group(1)
        else:
            filename = os.path.basename(file_path)
    else:
        filename = os.path.basename(file_path)
    
    logger.info(f"Converting file '{filename}' to source URL")
    
    # Remove the .md extension
    if filename.endswith(".md"):
        filename = filename[:-3]
    
    # Replace underscores with slashes to reconstruct the path
    path_parts = filename.split('_')
    
    # Special case for files from our website format "rte_fr_FAÉCUM_..."
    if len(path_parts) >= 3 and path_parts[0] == "rte" and path_parts[1] == "fr":
        # Reconstruct path as /rte/fr/FAÉCUM_...
        path = "/" + "/".join(path_parts[0:3])
        
        # Handle the rest of the parts which might contain underscores from the original URL
        if len(path_parts) > 3:
            path += "_" + "_".join(path_parts[3:])
            
        # Convert to proper URL format
        source_url = f"{BASE_URL}{path}"
        logger.info(f"DERIVED URL: '{source_url}' FROM: '{filename}'")
        return source_url
    
    # For other files, use a fallback URL with the filename
    source_url = f"{BASE_URL}/file/{filename}"
    logger.info(f"FALLBACK URL: '{source_url}' FROM: '{filename}'")
    return source_url


def load_documents(directory: Path = DATA_DIR) -> List[Document]:
    """Load documents from the specified directory."""
    logger.info(f"Loading documents from {directory}")
    
    if not directory.exists():
        logger.error(f"Directory {directory} does not exist")
        return []
    
    # Check if directory is empty
    files = list(glob.glob(f"{directory}/**/*.*", recursive=True))
    if not files:
        logger.warning(f"No files found in {directory}")
        return []
    
    # Load different file types 
    loaders = [
        DirectoryLoader(str(directory), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(directory), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(directory), glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
    ]
    
    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    
    # Manually fix the sources directly on each document
    for doc in documents:
        if 'source' in doc.metadata:
            file_path = doc.metadata['source']
            url = get_source_url_from_path(file_path)
            doc.metadata['source'] = url
            logger.info(f"SET DOC SOURCE: '{url}'")
        
    logger.info(f"Loaded {len(documents)} documents with proper source URLs")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents using recursive character text splitting."""
    if not documents:
        return []
    
    logger.info("Splitting documents into chunks")
    
    # Use a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Fix all chunk source URLs manually
    for chunk in chunks:
        if 'source' in chunk.metadata and chunk.metadata['source'].startswith(('data/', '/', './')):
            old_source = chunk.metadata['source']
            new_source = get_source_url_from_path(old_source)
            chunk.metadata['source'] = new_source
            logger.info(f"FIXED CHUNK SOURCE: '{old_source}' -> '{new_source}'")
    
    return chunks


def embed_and_store(chunks: List[Document], index_name: str = INDEX_NAME):
    """Embed chunks and store them in Pinecone."""
    if not chunks:
        logger.warning("No chunks to embed")
        return
    
    logger.info("Embedding and storing chunks in Pinecone")
    
    # Get Pinecone client
    pc = get_pinecone_client()
    if not pc:
        logger.error("Failed to initialize Pinecone client")
        return
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    
    # Final verification of chunks before storing
    logger.info("----- FINAL SOURCE URL VERIFICATION -----")
    for i, chunk in enumerate(chunks[:5]):
        src = chunk.metadata.get('source', 'NO SOURCE')
        logger.info(f"FINAL CHUNK #{i} SOURCE: '{src}'")
        if src.startswith(('data/', './', '/')):
            logger.error(f"ERROR: Chunk #{i} still has file path as source: '{src}'")
    
    try:
        # Store documents in Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name,
            text_key="text"
        )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in Pinecone")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error storing documents in Pinecone: {e}")
        return None


def main():
    """Main function to run the document ingestion process."""
    logger.info("Starting document ingestion process")
    
    # Clear existing vectors from Pinecone
    if not clear_pinecone_index():
        logger.warning("Failed to clear existing vectors, proceeding with ingestion anyway")
    
    # Load documents
    documents = load_documents()
    if not documents:
        logger.error("No documents to process")
        return
    
    # Chunk documents
    chunks = chunk_documents(documents)
    if not chunks:
        logger.error("No chunks created")
        return
    
    # Embed and store chunks
    embed_and_store(chunks)
    
    logger.info("Document ingestion process completed successfully")


if __name__ == "__main__":
    main() 