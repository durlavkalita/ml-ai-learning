from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import ollama
import os

# Load embedder (MiniLM: Fast; try 'all-mpnet-base-v2' for accuracy, but slower)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_FILE = 'quiz_index.faiss'

# Ingest function: Handles TXT/PDF, chunks with overlap
def ingest_notes(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        with open(file_path, 'r') as f:
            text = f.read()
    # Chunk: ~800 chars, 200 overlap (tweak for your notes!)
    chunk_size = 800
    overlap = 200
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Your notes here! E.g., history_notes.txt or startup_pitch.pdf
documents = ingest_notes('2312.10997v5.pdf')

if os.path.exists(INDEX_FILE):
    print("🚀 Local index found! Loading directly from disk...")
    index = faiss.read_index(INDEX_FILE)
else:
    print("⏳ No index found. Generating embeddings (this might take a moment)...")
    embeddings = embedder.encode(documents)
    embeddings = np.array(embeddings).astype('float32')

    print(f"Embedded {len(documents)} chunks!")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product: Better for relevance
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)  # Save for quick reloads
    print(f"✅ Embedded {len(documents)} chunks and saved index to '{INDEX_FILE}'!")

def retrieve(query, top_k=5, threshold=0.4):
    query_embedding = embedder.encode([query])[0].astype('float32')
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [(documents[i], distances[0][j]) for j, i in enumerate(indices[0]) if distances[0][j] > threshold]
    return [doc for doc, score in results] or ["No matches—expand your notes!"]

def generate_quiz(context, num_questions=5, style="multiple-choice"):
    if not context:
        return "No context found—try another topic!"
    prompt = f"Context: {' '.join(context)}\n\nCreate {num_questions} {style} quiz questions. For multiple-choice: 4 options, answer key at end. Make it engaging for learners!"
    response = ollama.generate(model='llama3.2', prompt=prompt) # llama3.1:8b for 8B param
    return response['response']

# Full flow: Your query here!
query = "What is rag?"  # Swap for your topic
retrieved = retrieve(query)
quiz = generate_quiz(retrieved, num_questions=4, style="true-false")
print("Retrieved:", retrieved)
print("\nQuiz:\n", quiz)