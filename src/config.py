import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class Config:
    # API
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MODEL = os.getenv("MODEL")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Path 
    ROOT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"
    DB_PATH = ROOT_DIR / "db"
    TEST_FILE_PATH = DATA_DIR / "C1" / "markdown" / "easy-rl-chapter1.md"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "BAAI/bge-m3"
    
    
if __name__ == "__main__":
    print("ROOT_DIR:" + Config.ROOT_DIR)
    print("model:" + Config.MODEL)