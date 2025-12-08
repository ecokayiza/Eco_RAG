from config import Config
from huggingface_hub import InferenceClient

###########################################
HF_TOKEN = Config.HF_TOKEN
MODEL_ID = Config.EMBEDDING_MODEL
###########################################

# Embedding via HuggingFace Inference API
class HuggingFaceEmbedder:
    @staticmethod
    def embed(data, model_id=MODEL_ID, hf_token=HF_TOKEN):
        try:
            client = InferenceClient(token=hf_token)
            embedding = client.feature_extraction(data, model=model_id)
            return embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None


if __name__ == "__main__":
    chunks = ["Hello, this is a test sentence for embedding."]
    embedding = HuggingFaceEmbedder.embed(chunks)
    if embedding:
        print(f"Success! Embedding dimension: {len(embedding[0])}")
        print(f"First 5 values: {embedding[0][:5]}")