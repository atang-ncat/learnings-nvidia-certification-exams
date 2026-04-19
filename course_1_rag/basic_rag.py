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
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("\n--- 2. BUILDING VECTOR DATABASE ---")
    # Embed and Store
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = FAISS.from_documents(chunks, embedder)
    
    # Create the Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("\n--- 3. ASSEMBLING THE AGENT ---")
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

    print("\n--- 4. EXECUTING INFERENCE ---")
    user_query = "What happens if the system gets too hot?"
    print(f"User Question: {user_query}\n")
    
    response = rag_chain.invoke({"input": user_query})
    print(f"Agent Answer: {response['answer']}")

if __name__ == "__main__":
    main()
