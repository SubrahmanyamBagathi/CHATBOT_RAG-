from typing import Any, Dict, List
from src.embedding import EmbeddingManager
from src.vector_store import VectorDB


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorDB, embedding_manager: EmbeddingManager):
        self.vector_store      = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query           : Search query string.
            top_k           : Number of top results to return.
            score_threshold : Minimum similarity score (0–1) to include a result.

        Returns:
            List of dicts with keys: id, content, metadata, similarity_score, rank.
        """
        print(f"Query: '{query}'  |  top_k={top_k}  |  threshold={score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                similarity = 1 - distance
                if similarity >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id"              : doc_id,
                            "content"         : document,
                            "metadata"        : metadata,
                            "similarity_score": similarity,
                            "distance"        : distance,
                            "rank"            : i + 1,
                        }
                    )

        print(f"Retrieved {len(retrieved_docs)} documents after filtering")
        return retrieved_docs