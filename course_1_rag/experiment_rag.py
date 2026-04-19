import os
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import numpy as np

def run_rag_experiment(chunk_size=100, chunk_overlap=20, k=2):
    print(f"=== EXPERIMENT: chunk_size={chunk_size}, overlap={chunk_overlap}, k={k} ===\n")
    
    file_path = "course_1_rag/data/hardware_manual.txt"
    
    # Load and Split
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:60]}... ({len(chunk.page_content)} chars)")

    # Embed and Store
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = FAISS.from_documents(chunks, embedder)
    
    # Create the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Initialize the LLM Brain
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

    # Set the System Prompt
    system_prompt = (
        "You are an expert hardware technician. "
        "Answer the question using ONLY the provided context. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Connect the chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # Execute inference
    user_query = "What happens if the system gets too hot?"
    print(f"\nUser Question: {user_query}")
    
    # Show what documents get retrieved
    retrieved_docs = vectorstore.similarity_search(user_query, k=k)
    print("Retrieved documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. {doc.page_content}")
    
    response = rag_chain.invoke({"input": user_query})
    print(f"\nAgent Answer: {response['answer']}\n")
    print("="*50 + "\n")

def main():
    # Run multiple experiments with different parameters
    print("RAG PARAMETER EXPERIMENTS\n")
    
    # Experiment 1: Small chunks, small overlap
    run_rag_experiment(chunk_size=50, chunk_overlap=10, k=2)
    
    # Experiment 2: Medium chunks, medium overlap (original)
    run_rag_experiment(chunk_size=100, chunk_overlap=20, k=2)
    
    # Experiment 3: Large chunks, large overlap
    run_rag_experiment(chunk_size=200, chunk_overlap=50, k=2)
    
    # Experiment 4: Same chunks but retrieve more documents
    run_rag_experiment(chunk_size=100, chunk_overlap=20, k=4)

if __name__ == "__main__":
    main()