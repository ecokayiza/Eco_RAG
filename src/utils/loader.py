import os
from config import Config
from abc import ABC, abstractmethod

###########################################
# Get Raw Data And Extract
###########################################

class DataLoader(ABC):
    """
    Abstract Base Class for all data loaders.
    Defines the contract for loading raw data.
    """
    @abstractmethod
    def load_data(self):
        """Load raw data from the source."""
        pass
    @abstractmethod
    def get_supported_extensions(self):
        """Return a list of supported file extensions."""
        pass

class MarkDownDataLoader(DataLoader):
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data
    def get_supported_extensions(self):
        return [".txt",'.md']

class PDFDataLoader(DataLoader):
    def load_data(self, file_path):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    def get_supported_extensions(self):
        return [".pdf"]

class DataLoaderFactory:
    """
    Factory class to create appropriate DataLoader instances based on file extension.
    """
    loaders = [MarkDownDataLoader(), PDFDataLoader()]
    
    @staticmethod
    def get_loader(file_extension):
        for loader in DataLoaderFactory.loaders:
            if file_extension in loader.get_supported_extensions():
                return loader
        raise ValueError(f"No loader found for extension: {file_extension}")

if __name__ == "__main__":

    try:
        file_path = Config.TEST_FILE_PATH
        _, ext = os.path.splitext(file_path)
        data = DataLoaderFactory.get_loader(ext).load_data(file_path)
        print(f"Loaded data from {file_path}:\n{data[:100]}...")  # Print first 100 characters
    except ValueError as e:
        print(e)