import numpy as np
from fastembed import TextEmbedding
from typing import List


class EmbeddingManager:
    """Handles document embedding generation using FastEmbed (no PyTorch/CUDA needed)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = TextEmbedding(model_name=self.model_name)
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = list(self.model.embed(texts))
        result = np.array(embeddings)
        print(f"Embeddings shape: {result.shape}")
        return result