import chromadb
import uuid
from typing import List, Dict, Any, Optional
from config import Config
import os

###########################################
# We use chromadb for vector storage
# This file defines basic db operations
DB_PATH = Config.DB_PATH
###########################################

class VectorDatabase:
    def __init__(self, collection_name: str = "rag_knowledge_base"):
        """
        Initialize connection to ChromaDB vector database
        """
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
        print(f"Connecting to VectorDB at: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        
        # metadata={"hnsw:space": "cosine"} for cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )

    def add_documents(self, 
                      texts: List[str], 
                      embeddings: List[List[float]], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None):
        """
        Add documents to the vector database
        """
        if not ids:
            # If no IDs are provided, generate random UUIDs
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        # Ensure metadatas is not None, ChromaDB requires a list
        if metadatas is None:
            metadatas = [{"source":"default"} for _ in range(len(texts))]

        try:
            self.collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully upserted {len(texts)} documents to collection.")
        except Exception as e:
            print(f"Error upserting documents: {e}")
    
    def query_by_metadata(self, where: Dict[str, Any], n_results: int = 5):
        """
        Query documents based on metadata filtering
        Args:
            where: Filter conditions, e.g., {"source_type": "pdf"}
            n_results: Number of similar chunks to retrieve
        Returns:
            A dictionary with query results, no distance included
            like {'document_ids': [...], 'documents': [...], 'metadatas': [...]
        """
        results = self.collection.get(
            where=where,
            limit=n_results
        )
        return results
    
    def query_with_vector(self, query_embedding: List[float], n_results: int = 5, where: Dict[str, Any] = None):
        """
        Query similar documents based on **one** query vector
        Args:
            query_embedding: The embedding vector to query
            n_results: Number of similar chunks to retrieve
            where: Filter conditions, e.g., {"source_type": "pdf"}
        Returns:
            A dictionary with query results
            like {'document_ids': [...], 'documents': [...], 'metadatas': [...], 'distances': [...]}
        """
        # since the query support list of vectors but we only need one
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where 
        )
        return results
    
    
    def delete_documents(self, where: Dict[str, Any]):
        """
        Delete documents matching the given metadata filter
        """
        try:
            # Get the documents that match the condition to count them
            results = self.collection.get(where=where)
            count = len(results['ids']) if results and 'ids' in results else 0
            
            if count > 0:
                self.collection.delete(where=where)
                print(f"Successfully deleted {count} documents matching {where}.")
            else:
                print(f"No documents found matching {where}.")
                
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def _clear_collection(self):
        """
        Delete all documents in the collection
        """
        try:
            collection_name = self.collection.name
            self.client.delete_collection(collection_name)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} 
            )
            print("Collection cleared.")
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def count(self):
        """
        Return the total number of documents in the collection
        """
        return self.collection.count()

    def peek(self, limit: int = 5):
        """
        Peek at a few documents in the collection
        """
        return self.collection.peek(limit=limit)
    


if __name__ == "__main__":
    try:
        vdb = VectorDatabase()
        # chunks = ["RAG 的核心是检索...", "向量数据库用于存储 Embedding..."]
        # from embedder import HuggingFaceEmbedder
        # embedder = HuggingFaceEmbedder()
        # embeddings = embedder.embed(chunks)

        # print("Current document count in DB:", vdb.count())
        # vdb._clear_collection()
        print("Current document count in DB:", vdb.count())

    except Exception as e:
        print(f"Test failed: {e}")
