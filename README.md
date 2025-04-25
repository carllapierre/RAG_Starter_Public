# Simple RAG Implementation with Langchain

This project provides a simple Retrieval Augmented Generation (RAG) implementation using LangChain/LangGraph, Langfuse, Pinecone, and OpenAI. It includes a Streamlit chat interface that supports conversation threads.

## Features

- Document ingestion
- Vector storage
- Chat interface with Streamlit
- Conversation thread support
- OpenAI for embeddings and LLM responses
- Contextual compression for improved retrieval

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key
- Langfuse API key
- Pinecone instance (cloud or local)

### Environment Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements-lock.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
4. Edit the `.env` file with your API keys and configuration

## Usage

### 1. Prepare Your Documents

Place your documents in the `./data/` directory. The following formats are supported:
- Text files (`.txt`)
- PDF files (`.pdf`)
- Markdown files (`.md`)

Create the data directory if it doesn't exist:
```bash
mkdir -p data
```

### 2. Ingest Documents

Run the ingestion script to process documents and store them in the vector database:
```bash
python ingest.py
```

This script will:
- Load documents from the `./data/` directory
- Split them into chunks
- Embed the chunks using OpenAI
- Store the embedded chunks in Pinecone

### 3. Run the Chat Application

Start the Streamlit chat application:
```bash
streamlit run app.py
```

This will open a web interface where you can:
- Ask questions about your documents
- Create new conversation threads
- Switch between different threads

## How It Works

### Document Ingestion

1. Documents are loaded from the `./data/` directory
2. The system splits the documents into meaningful chunks
3. OpenAI embeddings are used to embed these chunks
4. Embeddings are stored in a Vector Database for retrieval

Note: Dimensions are 3072 for OpenAI embeddings and the vector database requires 3072 dimensions.

### Question Answering

1. User questions are embedded using OpenAI
2. The system retrieves relevant chunks from the Vector Database
3. A contextual compression filter improves retrieval relevance
4. The LLM (OpenAI) generates responses based on the retrieved content and conversation history
