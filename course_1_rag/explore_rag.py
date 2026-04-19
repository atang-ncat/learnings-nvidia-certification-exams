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

def main():
    print("--- 1. SETTING UP DATA ---")
    file_path = "course_1_rag/data/hardware_manual.txt"
    
    # Create a temporary dummy file to load if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("The module operates between 15W and 60W. Overheating triggers the thermal throttle. Ensure heatsink is attached.")

    # Load and Split
    loader = TextLoader(file_path)
    docs = loader.load()
    
    print(f"Original document: {docs[0].page_content}")
    print(f"Document length: {len(docs[0].page_content)} characters\n")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content}")
        print(f"  Length: {len(chunk.page_content)} characters\n")

    print("\n--- 2. BUILDING VECTOR DATABASE ---")
    # Embed and Store
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = FAISS.from_documents(chunks, embedder)
    
    # Get the FAISS index and explore it
    index = vectorstore.index
    print(f"FAISS index type: {type(index)}")
    print(f"Number of vectors in index: {index.ntotal}")
    print(f"Dimension of each vector: {index.d}")
    
    # Get the actual vectors (embeddings)
    # FAISS stores vectors internally, we can reconstruct them
    embeddings = []
    for i in range(index.ntotal):
        # Reconstruct the vector at index i
        vector = index.reconstruct(i)
        embeddings.append(vector)
        print(f"Embedding {i+1} (first 5 dimensions): {vector[:5]}")
    
    embeddings_array = np.array(embeddings)
    print(f"\nAll embeddings shape: {embeddings_array.shape}")
    
    # Show similarity between chunks
    print("\n--- 3. SIMILARITY ANALYSIS ---")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            # Calculate cosine similarity
            dot_product = np.dot(embeddings[i], embeddings[j])
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            similarity = dot_product / (norm_i * norm_j)
            print(f"Similarity between chunk {i+1} and chunk {j+1}: {similarity:.4f}")

    # Create the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("\n--- 4. ASSEMBLING THE AGENT ---")
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

    print("\n--- 5. EXECUTING INFERENCE ---")
    user_query = "What happens if the system gets too hot?"
    print(f"User Question: {user_query}\n")
    
    # Let's also see what documents get retrieved
    retrieved_docs = vectorstore.similarity_search(user_query, k=2)
    print("Retrieved documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  {i+1}. {doc.page_content}")
    
    response = rag_chain.invoke({"input": user_query})
    print(f"\nAgent Answer: {response['answer']}")

if __name__ == "__main__":
    main()