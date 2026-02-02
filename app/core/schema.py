from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class AccessLevel(str, Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"

class DocumentType(str, Enum):
    POLICY = "policy"
    RFP = "rfp"
    SOP = "sop"

class ChunkMetadata(BaseModel):
    """
    Metadata schema for a single document chunk.
    Ensures governance attributes are always present.
    """
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: str
    document_type: DocumentType
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    access_level: AccessLevel
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
