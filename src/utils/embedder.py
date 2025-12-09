from config import Config
from huggingface_hub import InferenceClient
from tqdm import tqdm
###########################################
HF_TOKEN = Config.HF_TOKEN
MODEL_ID = Config.EMBEDDING_MODEL
###########################################

# Embedding via HuggingFace Inference API
class HuggingFaceEmbedder:
    @staticmethod
    def embed(data, model_id=MODEL_ID, hf_token=HF_TOKEN):
        """
        Embed chunks to vectors using HuggingFace Inference API
        data: list of text chunks
        """
        embeddings = []
        client = InferenceClient(model=model_id, token=hf_token)
        
        for chunk in tqdm(data, desc="Embedding chunks"):
            embedding = client.feature_extraction([chunk])
            if embedding is not None:
                embeddings.append(embedding[0])  # Assuming the first item is the embedding vector
            else:
                # retry 3 times
                retry_count = 0
                while retry_count < 3:
                    embedding = HuggingFaceEmbedder._embed_single([chunk], model_id, hf_token)
                    if embedding is not None:
                        embeddings.append(embedding[0])
                        break
                    retry_count += 1
                print(f"Failed to embed chunk after retries: {chunk[:30]}...")
                return None
        return embeddings

    @staticmethod
    def _embed_single(data, model_id, hf_token):
        client = InferenceClient(model=model_id, token=hf_token)
        return client.feature_extraction(data)


if __name__ == "__main__":
    file_path = Config.TEST_FILE_PATH
    import os
    from loader import DataLoaderFactory
    from chunker import ChunkerFactory
    data = DataLoaderFactory.load(file_path)
    _, ext = os.path.splitext(file_path)
    chunks = ChunkerFactory.chunk(data, ext)
    embeddings = HuggingFaceEmbedder.embed(chunks)