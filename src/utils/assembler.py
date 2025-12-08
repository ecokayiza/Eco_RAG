import json
import time
import os
from pathlib import Path    
from loader import DataLoaderFactory
from chunker import ChunkerFactory
from embedder import HuggingFaceEmbedder            
from vectorDatabase import VectorDatabase
from shcema import RAGRecord, RAGMetadata, ExtraAttributes
from config import Config


# This is the interface for outer files
# provide APIs to finish one whole process like:
# v  1.store_file:     filepath -> load -> chunk -> embed -> assemble records -> DB
# x  2.query:          query_text -> embed -> search in DB -> return results
# x  3. ....
##############################################
DATA_DIR = Config.DATA_DIR
##############################################

def get_relative_path(file_path,data_dir=DATA_DIR):
    # we use relative path
    try:
        rel_path = Path(file_path).relative_to(data_dir)
    except ValueError:
        # if file is not under DATA_DIR, use just the file name
        rel_path = Path(file_path).name
    return rel_path

class Assembler:
    
    @staticmethod
    def store_file(filepath):
        records = Assembler._get_records(filepath)
        db = VectorDatabase()
        Assembler._records_to_db(records,db)
    
    @staticmethod
    def _get_records(file_path):
        _,ext = os.path.splitext(file_path)
        data = DataLoaderFactory.get_loader(ext).load_data(file_path)
        chunks = ChunkerFactory.get_chunker(ext).chunk(data)
        print(f"Total {len(chunks)} chunks created.")
        embeddings = HuggingFaceEmbedder.embed(chunks)
        
        records = []
        for idx, chunk in enumerate(chunks):
 
            metadata = RAGMetadata(
                source_name=os.path.basename(file_path),
                source_type=ext.lstrip('.'),
                attributes=ExtraAttributes(
                    file_path=str(get_relative_path(file_path, DATA_DIR)).replace("\\", "/"), # 统一使用正斜杠
                    chunk_index=idx
                )
            )
            metadata.print_metadata()
            
            record = RAGRecord(
                document=chunk,
                metadata=metadata,
                vector=embeddings[idx]
            )
            records.append(record)
        return records
    
    @staticmethod
    def _records_to_db(records,db):
        # Batch Insert
        formatted_records = Assembler._records_to_db(records)
        
        if formatted_records:
            db.add_documents(
                ids=[r["id"] for r in formatted_records],
                texts=[r["document"] for r in formatted_records],
                embeddings=[r["vector"] for r in formatted_records],
                metadatas=[r["metadata"] for r in formatted_records]
            )
    
if __name__ == "__main__":
    file_path = Config.TEST_FILE_PATH
    Assembler.store_file(file_path)
    