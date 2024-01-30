from sqlalchemy import Column, Integer, String
from .database import Base

# Define SQLAlchemy model for the user table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    passport_url = Column(String, unique=True, index=True)