from beanie import Document
from pydantic import EmailStr, Field

class User(Document):
    email: EmailStr = Field(..., unique=True)
    hashed_password: str

    class Settings:
        name = "users"
