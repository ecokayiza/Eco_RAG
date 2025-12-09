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
    def chunk(data, file_extension):
        chunker = ChunkerFactory._get_chunker(file_extension)
        chunks = chunker.chunk(data)
        print(f"Total |{len(chunks)} chunks| created using {chunker.__name__}")
        return chunks
    
    @staticmethod
    def _get_chunker(file_extension):
        if file_extension in [".txt"]:
            return TextChunker
        elif file_extension in [".md"]:
            return MarkdownChunker
        else:
            return TextChunker  # Default to TextChunker



if __name__ == "__main__":
    
    import os
    from loader import DataLoaderFactory
    file_path = Config.TEST_FILE_PATH
    try:
        data = DataLoaderFactory.load(file_path)
        _, ext = os.path.splitext(file_path)
        chunks = ChunkerFactory.chunk(data, ext)        
    except ValueError as e:
        print(e)

