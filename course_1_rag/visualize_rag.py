import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def print_similarity_matrix(sim_matrix, chunk_texts):
    n = len(chunk_texts)
    # Print header
    print("    ", end="")
    for i in range(n):
        print(f"C{i+1:>2} ", end="")
    print()
    # Print rows
    for i in range(n):
        print(f"C{i+1:>2} ", end="")
        for j in range(n):
            val = sim_matrix[i, j]
            # Choose character based on similarity
            if val >= 0.9:
                ch = "█"
            elif val >= 0.8:
                ch = "▓"
            elif val >= 0.7:
                ch = "▒"
            elif val >= 0.6:
                ch = "░"
            elif val >= 0.5:
                ch = "·"
            else:
                ch = " "
            print(f"{ch:>2} ", end="")
        print(f"  ({val:.2f})" if j == n-1 else "")
    # Print legend
    print("\nLegend: █ >0.9 ▓ 0.8-0.9 ▒ 0.7-0.8 ░ 0.6-0.7 · 0.5-0.6 <0.5 space")

def main():
    file_path = "course_1_rag/data/hardware_manual.txt"
    
    # Load and Split
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks from document:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content}\n")
    
    # Embed
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = FAISS.from_documents(chunks, embedder)
    
    # Get vectors
    index = vectorstore.index
    n_vectors = index.ntotal
    dim = index.d
    print(f"Vector store: {n_vectors} vectors, each {dim}-dimensional\n")
    
    vectors = []
    for i in range(n_vectors):
        v = index.reconstruct(i)
        vectors.append(v)
    
    vectors = np.array(vectors)
    
    # Compute similarity matrix
    sim_matrix = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        for j in range(n_vectors):
            sim_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
    
    print("Cosine Similarity Matrix:")
    print_similarity_matrix(sim_matrix, [c.page_content for c in chunks])
    
    # Show vector norms
    norms = np.linalg.norm(vectors, axis=1)
    print("\nVector L2 norms:")
    for i, norm in enumerate(norms):
        print(f"  Chunk {i+1}: {norm:.4f}")
    
    # Attempt 2D visualization using PCA (if enough dimensions)
    if n_vectors >= 2 and dim >= 2:
        try:
            # Center the data
            mean_vec = np.mean(vectors, axis=0)
            centered = vectors - mean_vec
            # Covariance matrix
            cov = np.cov(centered, rowvar=False)
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # Project to first 2 PCs
            if eigvals[0] > 0 and eigvals[1] > 0:
                projection = centered @ eigvecs[:, :2]
                print("\n2D Projection (first two principal components):")
                for i, (x, y) in enumerate(projection):
                    print(f"  Chunk {i+1}: ({x:6.3f}, {y:6.3f})")
                # Show relative positions in a simple grid
                print("\nApproximate 2D layout (axes are PC1, PC2):")
                # Determine bounds
                x_min, x_max = projection[:,0].min(), projection[:,0].max()
                y_min, y_max = projection[:,1].min(), projection[:,1].max()
                # Add padding
                x_range = max(x_max - x_min, 0.1)
                y_range = max(y_max - y_min, 0.1)
                # Create a simple text grid
                grid_size = 20
                grid = [[' ' for _ in range(grid_size)] for __ in range(grid_size)]
                for i, (x, y) in enumerate(projection):
                    gx = int((x - x_min) / x_range * (grid_size-1))
                    gy = int((y - y_min) / y_range * (grid_size-1))
                    # Invert y for display (0 at top)
                    gy = grid_size - 1 - gy
                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        grid[gy][gx] = str(i+1)
                # Print grid
                print("   " + "".join(f"{i%10}" for i in range(grid_size)))
                for row_idx, row in enumerate(grid):
                    print(f"{row_idx%10:2} {''.join(row)}")
            else:
                print("\nNot enough variance for meaningful 2D projection.")
        except Exception as e:
            print(f"\nCould not compute 2D projection: {e}")
    else:
        print("\nNot enough vectors/dimensions for 2D projection.")

if __name__ == "__main__":
    main()