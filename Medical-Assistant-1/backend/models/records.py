from beanie import Document, Link
from pydantic import Field
from typing import Optional
from datetime import datetime
from models.user import User

class Record(Document):
    image_path: str
    analysis_result: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    owner: Link[User]

    class Settings:
        name = "records"
