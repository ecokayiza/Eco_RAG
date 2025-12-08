import json
import uuid
import time
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator

# === Metadata ===
class RAGMetadata(BaseModel):
    source_name: str
    source_type: str = "unknown"
    attributes: ExtraAttributes = Field(default_factory=ExtraAttributes) # Dynamic
    
    def print_metadata(self):
        print(f"Source Name: {self.source_name}")
        print(f"Source Type: {self.source_type}")
        print(f"Attributes: {self.attributes}")
        
# === ExtraAttributes ===
class ExtraAttributes(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunk_index: int = 0

# === A record ===
class RAGRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document: str
    vector: list[float] = None
    metadata: RAGMetadata

    def to_db_format(self) -> Dict[str, Any]:
        # put themt to json
        meta_dict = self.metadata.model_dump(exclude={"attributes"})
        attr_dict = self.metadata.attributes.model_dump(exclude_none=True)
        meta_dict["attributes_json"] = json.dumps(attr_dict, ensure_ascii=False)
        
        return {
            "id": self.id,
            "document": self.document,
            "vector": self.vector,
            "metadata": meta_dict
        }