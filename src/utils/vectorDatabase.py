import chromadb
import uuid
import json
import time
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
        初始化向量数据库客户端
        """
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH)
        print(f"Connecting to VectorDB at: {DB_PATH}")
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        
        # metadata={"hnsw:space": "cosine"} 指定使用余弦相似度
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
        向数据库添加文档和向量
        """
        if not ids:
            # 如果没有提供 ID，生成随机 UUID
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        # 确保 metadatas 不为 None，ChromaDB 需要列表
        if metadatas is None:
            metadatas = [{"source":"default"} for _ in range(len(texts))]

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(texts)} documents to collection.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def search(self, query_embedding: List[float], n_results: int = 5, where: Dict[str, Any] = None):
        """
        根据查询向量搜索相似文档
        
        Args:
            where: 过滤条件，例如 {"source_type": "pdf"}
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where 
        )
        return results

    def count(self):
        """
        返回集合中的文档数量
        """
        return self.collection.count()

    def peek(self, limit: int = 5):
        """
        查看前几条数据
        """
        return self.collection.peek(limit=limit)
    
    def delete_documents(self, ids: List[str]):
        """
        删除指定 ID 的文档
        """
        try:
            self.collection.delete(ids=ids)
            print(f"Successfully deleted {len(ids)} documents from collection.")
        except Exception as e:
            print(f"Error deleting documents: {e}")

if __name__ == "__main__":
    try:
        chunks = ["RAG 的核心是检索...", "向量数据库用于存储 Embedding..."]
        from embedder import HuggingFaceEmbedder
        embedder = HuggingFaceEmbedder()
        embeddings = embedder.embed(chunks)
        vdb = VectorDatabase()
        
        print("Current document count in DB:", vdb.count())
        vdb.add_documents(texts=chunks, embeddings=embeddings)
        print("Current document count in DB:", vdb.count())
        
        results = vdb.search(query_embedding=embeddings[0], n_results=2)
        print("Search results:", json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Test failed: {e}")
