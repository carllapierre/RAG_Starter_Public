#!/usr/bin/env python3
"""
RAG Chat Application

A Streamlit-based chat interface that uses:
- Pinecone for document retrieval
- LangGraph for a simple RAG flow
- OpenAI for embeddings and response generation
"""

import os
import uuid
import io
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated, Sequence

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse.callback import CallbackHandler
from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
import operator
from PIL import Image
import toml
import re

# Load environment variables
load_dotenv()

# Configure Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Langfuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about company_name based on the provided documents. 
If the answer is not in the documents, say you don't have enough information to answer the question. 
Don't make up information. Always cite your sources with the URLs from the documents.

company_description

ONLY CITE SOURCES IF YOU USE INFORMATION FROM THEM.
MAKE SURE IF THE USER IS SPEAKING FRENCH, YOU ANSWER IN FRENCH.
IF THE USER IS SPEAKING ENGLISH, YOU ANSWER IN ENGLISH.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())
    st.session_state.threads[st.session_state.current_thread_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "title": "New Conversation"
    }

# Define the state schema for our graph
class GraphState(TypedDict):
    messages: Annotated[Sequence[Dict], operator.add]
    question: str
    documents: Optional[List[Document]]
    answer: Optional[str]
    used_sources: Optional[List[str]]

# Pydantic model for retrieval decision
class RetrievalDecision(BaseModel):
    """Decision on whether to retrieve documents or not."""
    should_retrieve: bool = Field(description="Whether to retrieve documents from the knowledge base")
    reasoning: str = Field(description="Reasoning behind the decision")

def get_pinecone_client():
    """Create and return a Pinecone client."""
    try:
        # Initialize Pinecone client
        pc = Pinecone(
            api_key=PINECONE_API_KEY,
            host=PINECONE_HOST
        )
        
        # Try to describe the index to confirm connectivity
        try:
            pc_controller = Pinecone(api_key=PINECONE_API_KEY)
            index_description = pc_controller.describe_index(INDEX_NAME)
            return pc
            
        except Exception as e:
            st.error(f"Failed to access index '{INDEX_NAME}': {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize Pinecone client: {str(e)}")
        return None

def get_retriever(pc, index_name: str = INDEX_NAME):
    """Set up a retriever with Pinecone."""
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create vector store using Langchain's Pinecone wrapper
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Add embedding filter
    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.1
    )
    
    # Create contextual compression retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )
    
    return retriever

def create_rag_graph(retriever, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Create a simple LangGraph for RAG."""
    # Initialize language model
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2
    )
    
    # Define the graph
    workflow = StateGraph(GraphState)
    
    # Node 1: Decide if we need to retrieve documents
    def should_retrieve(state: GraphState) -> Dict:
        """Determine if we need to retrieve documents."""
        # Convert list of dict messages to LangChain message types
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        # Create a prompt to determine if we need to retrieve
        retrieval_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You determine if a user question requires retrieving information from a knowledge base.
            
            Set 'should_retrieve' to true if:
            - The question asks for specific information
            - The question requires factual information
            
            Set 'should_retrieve' to false if:
            - The question is conversational or about general capabilities
            - The question is asking for clarification about previous messages
            - The question can be answered with general knowledge
            
            Respond with JSON: {"should_retrieve": true/false, "reasoning": "explanation"}
            """),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{question}")
        ])
        
        # Get the decision
        chain = retrieval_prompt | llm
        
        try:
            response = chain.invoke({
                "history": messages[-5:] if len(messages) > 5 else messages,
                "question": state["question"]
            })
            
            # Parse the response
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```|{.*}', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(0)
                parsed_json = json.loads(json_str)
                
                # Return the next node based on the decision
                if parsed_json.get("should_retrieve", True):
                    return {"next": "retrieve_and_generate"}
                else:
                    return {"next": "generate_without_rag"}
            else:
                # Default to retrieval if we can't parse the response
                return {"next": "retrieve_and_generate"}
                
        except Exception as e:
            print(f"Error in should_retrieve: {str(e)}")
            # Default to retrieval if there's an error
            return {"next": "retrieve_and_generate"}
    
    # Node 2: Retrieve documents and generate response with RAG
    def retrieve_and_generate(state: GraphState) -> Dict:
        """Retrieve relevant documents and generate a response."""
        # Convert messages for context
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        # Retrieve documents
        print(f"Retrieving documents for: {state['question']}")
        docs = retriever.invoke(state["question"])
        
        # Print document sources for debugging
        print("\n----- RETRIEVED DOCUMENTS -----")
        used_sources = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            print(f"Document #{i+1} source: {source}")
            used_sources.append(source)
        print("------------------------------\n")
        
        # Format documents for the prompt
        context = ""
        if docs:
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown")
                text = doc.page_content
                formatted_docs.append(f"Document {i+1} [Source URL: {source}]\n{text}\n")
            context = "\n".join(formatted_docs)
        
        # Create a prompt for the RAG response
        rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            {system_prompt}
            
            Do NOT include citations inline in your response. Just provide a clear, helpful answer.
            
            Keep track of which documents you use to answer the question. After your response, 
            I will add the sources automatically in a separate section.
            
            If you don't find relevant information in the documents, say so honestly.
            """),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{question}" + (f"\n\nRelevant documents:\n{context}" if context else ""))
        ])
        
        # Generate the response
        chain = rag_prompt | llm
        
        try:
            response = chain.invoke({
                "history": messages[-5:] if len(messages) > 5 else messages,
                "question": state["question"]
            })
            
            # Create a formatted response with sources at the bottom
            if docs:
                # Filter to only include relevant sources (remove duplicates)
                relevant_sources = []
                for source in used_sources:
                    if source not in relevant_sources and source != "Unknown" and source != "#":
                        relevant_sources.append(source)
                
                if relevant_sources:
                    # Create HTML for the source links
                    source_links = []
                    for i, source in enumerate(relevant_sources):
                        # Extract domain for display
                        domain = source.split('/')[2] if len(source.split('/')) > 2 else source
                        source_links.append(f'<a href="{source}" target="_blank" class="source-link">Source {i+1}: {domain}</a>')
                    
                    # Add source section
                    sources_html = "<div class='sources-section'><strong>Sources:</strong><br>" + "<br>".join(source_links) + "</div>"
                    full_response = f"{response.content}\n\n{sources_html}"
                else:
                    full_response = response.content
            else:
                full_response = response.content
            
            return {
                "answer": full_response,
                "documents": docs,
                "used_sources": used_sources
            }
            
        except Exception as e:
            print(f"Error generating RAG response: {str(e)}")
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "documents": [],
                "used_sources": []
            }
    
    # Node 3: Generate response without RAG
    def generate_without_rag(state: GraphState) -> Dict:
        """Generate a response without retrieving documents."""
        # Convert messages for context
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        # Create a prompt for the direct response
        direct_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{question}")
        ])
        
        # Generate the response
        chain = direct_prompt | llm
        
        try:
            response = chain.invoke({
                "history": messages[-5:] if len(messages) > 5 else messages,
                "question": state["question"]
            })
            
            return {
                "answer": response.content,
                "documents": [],
                "used_sources": []
            }
            
        except Exception as e:
            print(f"Error generating direct response: {str(e)}")
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "documents": [],
                "used_sources": []
            }
    
    # Add nodes to the graph
    workflow.add_node("should_retrieve", should_retrieve)
    workflow.add_node("retrieve_and_generate", retrieve_and_generate)
    workflow.add_node("generate_without_rag", generate_without_rag)
    
    # Add edges
    workflow.add_edge(START, "should_retrieve")
    workflow.add_conditional_edges(
        "should_retrieve",
        lambda x: x["next"],
        {
            "retrieve_and_generate": "retrieve_and_generate",
            "generate_without_rag": "generate_without_rag"
        }
    )
    workflow.add_edge("retrieve_and_generate", END)
    workflow.add_edge("generate_without_rag", END)
    
    # Compile the graph
    return workflow.compile()

def create_new_thread():
    """Create a new conversation thread."""
    thread_id = str(uuid.uuid4())
    st.session_state.current_thread_id = thread_id
    st.session_state.threads[thread_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "title": "New Conversation"
    }
    st.session_state.messages = []

def switch_thread(thread_id: str):
    """Switch to an existing thread."""
    if thread_id in st.session_state.threads:
        st.session_state.current_thread_id = thread_id
        st.session_state.messages = st.session_state.threads[thread_id]["messages"]

def save_thread(thread_id: str, messages: List[Dict]):
    """Save messages to a thread."""
    if thread_id in st.session_state.threads:
        st.session_state.threads[thread_id]["messages"] = messages
        
        # Update title if this is the first user message
        if len(messages) == 2 and "thread_title" not in st.session_state.threads[thread_id]:
            first_msg = messages[0]["content"] if messages else ""
            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
            st.session_state.threads[thread_id]["title"] = title
            st.session_state.threads[thread_id]["thread_title"] = True

def preprocess_content(content):
    """Preprocess content to prevent markdown issues."""
    if '\\' in content:
        return content
        
    content = re.sub(r'(\w+)_(\w+)', r'\1\_\2', content)
    content = content.replace('$', '<span>$</span>')
    content = content.replace('*', '\\*')
    
    return content

def render_message(message):
    """Render a chat message."""
    if message["role"] == "assistant":
        try:
            img = Image.open("assets/chat-icon.png")
            img = img.resize((38, 38))
            if img.mode in ('P', 'RGBA'):
                img = img.convert('RGB')
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            
            with st.chat_message(message["role"], avatar=buf.getvalue()):
                # Don't preprocess URLs in content as we're handling them specifically
                content = message["content"]
                # Use unsafe_allow_html=True to render the links properly
                st.write(f'<div class="assistant-message" data-tex="false">{content}</div>', unsafe_allow_html=True)
        except Exception as e:
            print(f"Error setting custom avatar: {e}")
            with st.chat_message(message["role"]):
                content = message["content"]
                st.write(f'<div class="assistant-message" data-tex="false">{content}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    """Main function to run the Streamlit app."""
    config = toml.load(".streamlit/config.toml")

    # Custom CSS
    st.markdown("""
        <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #fafafa;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Sidebar button styling */
        [data-testid="stSidebar"] button {
            background-color: white !important;
            color: #925cff !important;
            border: 1px solid #925cff !important;
            border-radius: 4px !important;
        }
        
        /* Sidebar button hover state */
        [data-testid="stSidebar"] button:hover {
            background-color: #925cff !important;
            color: white !important;
        }

        /* Ensure white background for all input-related elements */
        .stTextInput, 
        .stTextInput > div, 
        textarea,
        .stChatInput {
            background-color: white !important;
        }
                
        /*  nice font */
        body {
            font-family: 'Arial', sans-serif;
        }
        
        /* Sources section styling */
        .sources-section {
            margin-top: 20px;
            padding: 12px;
            border-top: 1px solid #e0e0e0;
            background-color: #f8f9fa;
            border-radius: 0 0 8px 8px;
        }
        
        .sources-section strong {
            display: block;
            margin-bottom: 8px;
            color: #333;
        }
        
        .source-link {
            text-decoration: none;
            color: rgb(35, 91, 168);
            padding: 2px 6px;
            margin: 4px 2px;
            border-radius: 4px;
            background-color: rgba(35, 91, 168, 0.1);
            transition: all 0.3s ease;
            font-weight: 500;
            display: inline-block;
        }
        
        .source-link:hover {
            background-color: rgba(35, 91, 168, 0.2);
            text-decoration: underline;
        }
        
        /* Assistant message styling */
        .assistant-message {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            white-space: pre-line;
        }

        /* Disable math styling */
        .assistant-message .katex,
        .assistant-message .katex-mathml,
        .assistant-message .katex-html {
            display: none !important;
        }

        .assistant-message .math-inline {
            font-family: inherit !important;
            font-style: normal !important;
        }
        
        /* User avatar color */
        [data-testid="stChatMessageAvatar"][data-title="user"] {
            background-color: rgb(35, 91, 168) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display title
    title = config["app"]["title"]
    st.title(title)
    
    # Initialize Langfuse for tracing
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST
    )
    
    # Create Langfuse callback handler
    langfuse_handler = CallbackHandler(
        secret_key=LANGFUSE_SECRET_KEY,
        public_key=LANGFUSE_PUBLIC_KEY,
        host=LANGFUSE_HOST
    )
    
    # Sidebar for thread management
    with st.sidebar:
        # Display logo in sidebar
        try:
            st.image("assets/logo.png", width=80)
        except Exception:
            st.write("🤖")
        st.title("Conversations")
        
        # Button to create new thread
        if st.button("Nouvelle conversation"):
            create_new_thread()
            st.rerun()
        
        # List existing threads
        st.divider()
        st.subheader("Vos conversations")
        
        # Sort threads by creation time (newest first)
        sorted_threads = sorted(
            st.session_state.threads.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        
        for thread_id, thread_data in sorted_threads:
            thread_title = thread_data.get("title", "Untitled")
            if st.button(
                thread_title,
                key=f"thread_{thread_id}",
                use_container_width=True,
                type="secondary" if thread_id == st.session_state.current_thread_id else "primary"
            ):
                switch_thread(thread_id)
                st.rerun()
    
    try:
        # Initialize Pinecone client
        pc = get_pinecone_client()
        
        # Set up retriever
        retriever = get_retriever(pc)
        if not retriever:
            st.warning("Please upload documents using the ingestion script first.")
            return
        
        # Set up conversation chain with LangGraph
        chain = create_rag_graph(retriever=retriever)
        
        # Display chat messages
        for message in st.session_state.messages:
            render_message(message)
        
        # Chat input
        if prompt := st.chat_input("Demandez une question..."):
            # Add user message to session
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_message({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                # Prepare the initial state for the graph
                initial_state = {
                    "messages": st.session_state.messages,
                    "question": prompt,
                    "documents": None,
                    "answer": None,
                    "used_sources": None
                }
                
                # Invoke the graph with Langfuse callback
                result = chain.invoke(
                    initial_state, 
                    config={"callbacks": [langfuse_handler]}
                )
                
                # Add AI response to session
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                render_message({"role": "assistant", "content": result["answer"]})
            
            # Save updated messages to current thread
            save_thread(st.session_state.current_thread_id, st.session_state.messages)
    
    finally:
        # Ensure Langfuse client is properly closed
        langfuse.flush()

if __name__ == "__main__":
    main() 