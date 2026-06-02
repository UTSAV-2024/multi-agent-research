from motor.motor_asyncio import AsyncIOMotorClient

from app.config.settings import settings

client = AsyncIOMotorClient(
    settings.MONGODB_URL
)

database = client[
    settings.DATABASE_NAME
]


def get_database():

    return database