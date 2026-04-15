import os
import uuid
import numpy as np
import chromadb
from typing import Any, List
from langchain.schema import Document


class VectorDB:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "datasource",
        persist_directory: str = "vector_store",   # relative path, cloud-friendly
    ):
        self.collection_name   = collection_name
        self.persist_directory = persist_directory
        self.client            = None
        self.collection        = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )

            print(f"Vector store ready  |  collection: '{self.collection_name}'")
            print(f"Existing chunks     : {self.collection.count()}")
        except Exception as e:
            raise RuntimeError(f"Error initializing vector store: {e}")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents : List of LangChain Document objects.
            embeddings: Corresponding embedding array (same length).
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        ids, embedding_list, metadatas, texts = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"]      = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            texts.append(doc.page_content)
            embedding_list.append(emb.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=texts,
            )
            print(f"Added {len(documents)} chunks  |  total: {self.collection.count()}")
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to vector store: {e}")