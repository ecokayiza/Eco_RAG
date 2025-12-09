import json
import uuid
import time
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator

# === ExtraAttributes ===
class ExtraAttributes(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunk_index: int = 0

# === Metadata ===
class RAGMetadata(BaseModel):
    source_name: str
    source_type: str = "unknown"
    attributes: ExtraAttributes = Field(default_factory=ExtraAttributes) # Dynamic
    
    def print_metadata(self):
        print(f"    Source Name: {self.source_name}")
        print(f"    Source Type: {self.source_type}")
        print(f"    Attributes: {self.attributes}")

# === A record ===
class RAGRecord(BaseModel):
    id: Optional[str] = None
    document: str
    vector: Optional[list[float]] = None
    distance: Optional[float] = None
    metadata: RAGMetadata
    
    @model_validator(mode='after')
    def generate_deterministic_id(self):
        """
        Automatically generate a deterministic ID based on metadata if ID is missing.
        """
        if self.id is None:
            unique_id_str = f"{self.metadata.source_name}_{self.metadata.source_type}_{self.metadata.attributes.file_path}_{self.metadata.attributes.chunk_index}"
            self.id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_id_str))
        return self

    def to_db_format(self) -> Dict[str, Any]:
        # put themt to json
        meta_dict = self.metadata.model_dump(exclude={"attributes"})
        attr_dict = self.metadata.attributes.model_dump(exclude_none=True)
        # Flatten attributes into metadata for easier querying
        meta_dict.update(attr_dict)
        
        return {
            "id": self.id,
            "document": self.document,
            "vector": self.vector,
            "metadata": meta_dict
        }
        
    def get_id(self):
        return self.id    

    def print(self):
        print(f"{'-'*20} RAG Record {'-'*20}")
        print(f"ID: {self.id}")
        print(f"Document:\n {self.document[:50]}...")  # Print first 50 chars
        if self.vector:
            print(f"Vector (first 5 dims): {self.vector[:5]}...")
        if self.distance is not None:
            print(f"Distance: {self.distance}")
        print("Metadata:")
        self.metadata.print_metadata()
        print(f"{'-'*52}")

    @staticmethod
    def get_records_from_results(results: Dict[str, Any]) -> list['RAGRecord']:
        """
        Transform query results from vector DB back to RAGRecord objects
        *note: vector is not included in results, we use distance instead*
        """
        records = []
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        distances = results.get('distances', [])
        ids = results.get('ids', [])

        # Handle ChromaDB's list of lists format (if query returns multiple results lists)
        if documents and isinstance(documents[0], list):
            documents = documents[0]
            metadatas = metadatas[0] if metadatas else []
            distances = distances[0] if distances else []
            ids = ids[0] if ids else []

        for idx, doc in enumerate(documents):
            metadata_dict = metadatas[idx] if idx < len(metadatas) else {}
            record_id = ids[idx] if idx < len(ids) else None
            record_distance = distances[idx] if idx < len(distances) else None

            # Reconstruct ExtraAttributes
            attributes = ExtraAttributes(
                file_path=metadata_dict.pop('file_path', None),
                url=metadata_dict.pop('url', None),
                chunk_index=metadata_dict.pop('chunk_index', 0)
            )
            metadata = RAGMetadata(
                source_name=metadata_dict.get('source_name', 'unknown'),
                source_type=metadata_dict.get('source_type', 'unknown'),
                attributes=attributes
            )
            record = RAGRecord(
                id=record_id, # Use the ID from DB
                document=doc,
                distance=record_distance,
                metadata=metadata
            )
            records.append(record)
        return records

    @staticmethod
    def sort_by_distance(records: list['RAGRecord']) -> list['RAGRecord']:
        """
        Sort a list of RAGRecords by their distance attribute in ascending order
        **note: only records transformed from results queried with vectors that have distance**
        you probably dont need this since db already returns sorted results
        """
        return sorted(records, key=lambda r: r.distance if r.distance is not None else float('inf'))
    
if __name__ == "__main__":
    # Test RAGRecord creation and ID generation
    metadata = RAGMetadata(
        source_name="test_source",
        source_type="txt",
        attributes=ExtraAttributes(
            file_path="/path/to/file.txt",
            chunk_index=1
        )
    )
    record = RAGRecord(
        document="This is a test document chunk.",
        vector=[0.1, 0.2, 0.3],
        metadata=metadata
    )
    record.print()
    
    print("DB Format:\n", record.to_db_format())