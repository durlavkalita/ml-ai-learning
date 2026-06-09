from typing import List, Dict
import re
import numpy as np

class RAG():
    def __init__(self, filepath:str = "policy.txt"):
        self.filepath = filepath
        self.chunks: List[str] = []
        self.vocab_list: List[str] = []
        self.idf_vector: np.ndarray = np.array([])
        self.tf_idf_matrix: np.ndarray = np.array([])

    def create_chunks(self, chunk_size: int = 50, overlap: int = 10) -> List[str]:
        """
        Create chunks list from the provided document.
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size should be less than overlap.")
        
        with open(self.filepath, 'r') as f:
            data = f.read()
        
        words = [word for word in re.split(r'\s+', data) if word]

        chunks: List[str] = []
        
        for i in range(0, len(words), chunk_size-overlap):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        self.chunks = chunks
        return chunks
    
    def create_vocab(self) -> List[str]:
        """
        Create vocabulary across all chunks. In each chunks the number of unique elements may vary. Applying cosine similarity later this vocab will be used. for every chunk tf vector will be same length with absent word having 0.0 as value 
        """
        vocab_set: set[str] = set()
        for chunk in self.chunks:
            words = [w.lower() for w in re.split(r'\s+', chunk) if w]
            vocab_set.update(words)

        self.vocab_list = sorted(list(vocab_set))
        return self.vocab_list
  
    def create_tf_idf_matrix(self):
        """
        Create tf vector - for each chunk, count how often each word appears ÷ total words in chunk
        Create idf vector - log(total_chunks / chunks_containing_term) — penalises common words
        """
        tokenized_chunks: List[List[str]] = []
        for chunk in self.chunks:
            words = [w.lower() for w in re.split(r'\s+', chunk) if w]
            tokenized_chunks.append(words)

        tf_matrix_list: List[List[float]] = []
        for words in tokenized_chunks:
            total_words_in_chunk = len(words)

            if total_words_in_chunk == 0:
                current_chunk_tf = [0.0]*len(self.vocab_list)
                tf_matrix_list.append(current_chunk_tf)
                continue

            local_counts: Dict[str,int] = {}
            for word in words:
                local_counts[word] = local_counts.get(word, 0) + 1

            current_chunk_tf: List[float] = []
            for vocab_word in self.vocab_list:
                word_count = local_counts.get(vocab_word, 0)
                current_chunk_tf.append(word_count/total_words_in_chunk)

            tf_matrix_list.append(current_chunk_tf)
            
        tf_matrix = np.array(tf_matrix_list, dtype=np.float64)

        total_chunks = len(tokenized_chunks)
        chunks_containing_term = np.sum(tf_matrix>0, axis=0)
        self.idf_vector = np.log(total_chunks / (1 + chunks_containing_term))

        self.tf_idf_matrix = tf_matrix * self.idf_vector
        return self.tf_idf_matrix

    def query_vectorization(self, query: str):
        """
        Transforms a runtime query string into an aligned 1D TF-IDF vector.
        """
        # clean and tokenize single query string
        query_words = [word.lower() for word in re.split(r'\s+', query) if word]
        total_words = len(query_words)
        
        if total_words == 0:
            return np.zeros(len(self.vocab_list), dtype=np.float64)
        
        query_counts: Dict[str, int] = {}
        for word in query_words:
            query_counts[word] = query_counts.get(word, 0) + 1

        query_tf: List[float] = []
        for vocab_word in self.vocab_list:
            word_count = query_counts.get(vocab_word, 0)
            query_tf.append(word_count / total_words)

        query_tf_vector = np.array(query_tf, dtype=np.float64)

        query_tf_idf_vector = query_tf_vector * self.idf_vector

        return query_tf_idf_vector
    
    def search(self, query: str, top_k: int = 3):
        """
        Computes cosine similarities across the entire database matrix at once.
        Returns sorted tracking lists containing tuples of (chunk_index, score).
        """
        query_vec = self.query_vectorization(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return [(i, 0.0) for i in range(min(top_k, len(self.chunks)))]
        
        matrix_norms = np.linalg.norm(self.tf_idf_matrix, axis=1)
        # Handle zero-norm fallback for unpopulated document entries safely
        matrix_norms[matrix_norms == 0] = 1.0

        # Matrix-vector dot product computes all intersections simultaneously
        dot_products = np.dot(self.tf_idf_matrix, query_vec)
        similarities = dot_products / (matrix_norms * query_norm)

        # Extract the highest alignment indexes sorted in descending order
        ranked_indices = np.argsort(similarities)[::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in ranked_indices[:top_k]]
    
    def fit(self, chunk_size: int = 50, overlap: int = 10):
        """
        High-level orchestration method initializing total pipeline state indexing.
        """
        self.create_chunks(chunk_size, overlap)
        self.create_vocab()
        self.create_tf_idf_matrix()

if __name__ == '__main__':
    rag = RAG()
    rag.fit()

    query = "GENERATIVE AI CODE QUALITY PROTOCOLS"
    results = rag.search(query, top_k=3)

    print(f"=== Search Results for: '{query}' ===\n")
    
    # Iterate through the returned tuples of (chunk_index, relevance_score)
    for rank, (idx, score) in enumerate(results, start=1):
        # Extract the string snippet from our memory store using the index integer
        matched_text = rag.chunks[idx]
        
        print(f"Result Rank #{rank}")
        print(f"Chunk Index: {idx} | Cosine Similarity Score: {score:.4f}")
        print(f"Excerpt: \"{matched_text}\"")
        print("-" * 50)