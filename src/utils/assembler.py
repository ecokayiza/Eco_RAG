import os
import uuid
from config import Config
from shcema import RAGRecord, RAGMetadata, ExtraAttributes
from loader import DataLoaderFactory
from chunker import ChunkerFactory
from embedder import HuggingFaceEmbedder            
from vectorDatabase import VectorDatabase



# This is the interface for outer files
# provide APIs to finish one whole process like:
# v  1.store_file:     filepath -> load -> chunk -> embed -> assemble records -> DB
# x  2.query_file:     filepath -> query file in DB -> return results
# x  3.delete_file:    filepath -> find records in DB -> delete records
##############################################

class Assembler:
    db = VectorDatabase()
    
    @staticmethod
    def store_file(filepath):
        """
        Store file to vdb.
        filepath: should be absolute path.
        """
        records = Assembler._get_records(filepath)
        Assembler._records_to_db(records, Assembler.db)
    
    @staticmethod
    def delete_file(filepath):
        """
        Delete file from vdb.
        filepath: should be absolute path.
        """
        rel_path = Config.get_relative_path(filepath)
        where = {"file_path": rel_path}
        Assembler.db.delete_documents(where)
        
    @staticmethod
    def query_file(filepath):
        """
        Query file from vdb.
        Args:
            filepath: should be absolute path.
        Returns:
            A dictionary with query results, *no distance included* since we query by filters
            like {'document_ids': [...], 'documents': [...], 'metadatas': [...]}
        """
        rel_path = Config.get_relative_path(filepath)
        where = {"file_path": rel_path}
        results = Assembler.db.query_by_metadata(where, n_results=99999999)
        print(f"Found {len(results.get('documents', []))} documents for file: {rel_path}")
        return results
    
    @staticmethod
    def query_with_vector(vector, n_results=5, where=None):
        """
        Query similar documents based on **one** query vector
        """
        results = Assembler.db.query_with_vector(vector, n_results, where)
        # results['documents'] is a list of lists, so we check the length of the first list
        doc_count = len(results.get('documents', [[]])[0])
        print(f"Found {doc_count} documents for the query vector.")
        return results
    
    @staticmethod
    def _get_records(file_path):
        """
        Get records objects from file_path, to create a record, we need:
            chunk text, embedding vector, metadata
        """
        _,ext = os.path.splitext(file_path)
        data = DataLoaderFactory.load(file_path)
        chunks = ChunkerFactory.chunk(data, ext)
        embeddings = HuggingFaceEmbedder.embed(chunks)
        
        records = []
        for idx, chunk in enumerate(chunks):
 
            metadata = RAGMetadata(
                source_name=os.path.basename(file_path),
                source_type=ext.lstrip('.'),
                attributes=ExtraAttributes(
                    file_path=Config.get_relative_path(file_path),
                    chunk_index=idx
                )
            )
            # metadata.print_metadata()
            
            record = RAGRecord(
                document=chunk,
                metadata=metadata,
                vector=embeddings[idx]
            )
            records.append(record)
        return records
    
    @staticmethod
    def _records_to_db(records,db):
        """
        Insert records to vector DB
        """
        formatted_records = [r.to_db_format() for r in records]
        
        if formatted_records:
            db.add_documents(
                ids=[r["id"] for r in formatted_records],
                texts=[r["document"] for r in formatted_records],
                embeddings=[r["vector"] for r in formatted_records],
                metadatas=[r["metadata"] for r in formatted_records]
            )
    
if __name__ == "__main__":
    file_path = Config.TEST_FILE_PATH
    
    # print("Dpocument count in DB before storing file:", Assembler.db.count())
    # Assembler.store_file(file_path)
    # print("Document count in DB after storing file:", Assembler.db.count())
    
    # print("Document count in DB before deleting file:", Assembler.db.count())
    # Assembler.delete_file(file_path)
    # print("Document count in DB after deleting file:", Assembler.db.count())
    
    # results = Assembler.query_file(file_path)
    # records = RAGRecord.get_records_from_results(results)
    
    text = ["智能体采取的动作"]
    embedding = HuggingFaceEmbedder.embed(text)[0]
    results = Assembler.query_with_vector(embedding, n_results=5)
    records = RAGRecord.get_records_from_results(results)
    # records = RAGRecord.sort_by_distance(records)
    
    for record in records:
        record.print()


     