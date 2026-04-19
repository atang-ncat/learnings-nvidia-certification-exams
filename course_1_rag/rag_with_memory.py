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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Global store for message histories (in practice, you'd use a database)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def main():
    print("--- 1. SETTING UP DATA ---")
    file_path = "course_1_rag/data/hardware_manual.txt"
    
    # Create a temporary dummy file to load if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("The module operates between 15W and 60W. Overheating triggers the thermal throttle. Ensure heatsink is attached. The recommended operating temperature range is 0°C to 85°C. Power consumption varies with workload, peaking at 60W under heavy load. Proper ventilation is essential to maintain performance and prevent thermal throttling.")

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

    print("\n--- 3. ASSEMBLING THE RAG CHAIN WITH MEMORY ---")
    # Initialize the LLM Brain
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

    # Create a prompt that includes conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert hardware technician. Answer the question using ONLY the provided context. Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),  # This is where history goes
        ("human", "{input}"),
    ])

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("\n--- 4. EXECUTING CONVERSATIONAL INFERENCE ---")
    
    # Simulate a conversation
    session_id = "user_session_1"
    
    # First question
    print("User Question 1: What happens if the system gets too hot?")
    response1 = conversational_rag_chain.invoke(
        {"input": "What happens if the system gets too hot?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Agent Answer 1: {response1['answer']}\n")
    
    # Second question - should remember context from first
    print("User Question 2: What should I do to prevent this?")
    response2 = conversational_rag_chain.invoke(
        {"input": "What should I do to prevent this?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Agent Answer 2: {response2['answer']}\n")
    
    # Third question - referencing earlier conversation
    print("User Question 3: You mentioned thermal throttle earlier, what triggers it?")
    response3 = conversational_rag_chain.invoke(
        {"input": "You mentioned thermal throttle earlier, what triggers it?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Agent Answer 3: {response3['answer']}\n")
    
    # Show the conversation history
    print("--- 5. CONVERSATION HISTORY ---")
    history = get_session_history(session_id)
    print(f"Number of messages in history: {len(history.messages)}")
    for i, msg in enumerate(history.messages):
        if isinstance(msg, HumanMessage):
            print(f"  Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"  AI: {msg.content}")

if __name__ == "__main__":
    main()