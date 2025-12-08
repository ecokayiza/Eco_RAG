import os
from config import Config
from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter

###########################################
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP
###########################################

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        pass

class TextChunker(Chunker):
    @staticmethod
    def chunk(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", ",", "，", " "]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
class MarkdownChunker(Chunker):
    @staticmethod
    def chunk(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "#", "##", "###", ".", "。", ",", "，", " "]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
class ChunkerFactory:
    @staticmethod
    def get_chunker(file_extension):
        if file_extension in [".txt"]:
            return TextChunker
        elif file_extension in [".md"]:
            return MarkdownChunker
        else:
            return TextChunker  # Default to TextChunker



if __name__ == "__main__":
    
    from loader import DataLoaderFactory
    def chunk(file):
        _, ext = os.path.splitext(file)
        data = DataLoaderFactory.get_loader(ext).load_data(file)
        chunks = ChunkerFactory.get_chunker(ext).chunk(data)
        return chunks

    file_path = Config.TEST_FILE_PATH
    try:
        chunks = chunk(file_path)
        print(f"File: {file_path} | Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):  # Print first 2 chunks as sample
            print(f"--- Chunk {i+1} ---\n{chunk}\n")
        
    except ValueError as e:
        print(e)

