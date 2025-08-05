import motor.motor_asyncio
from beanie import init_beanie
from models.user import User
from models.records import Record

# Replace with your MongoDB connection string
MONGO_DETAILS = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.medbud # Database name

async def init_db():
    # Initialize beanie with the User document
    # We will add more documents here as we create them
    await init_beanie(database=database, document_models=[User, Record])
