from sqlalchemy import Column, Integer, String
from .database import Base

# Define SQLAlchemy model for the user table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    passport_url = Column(String, unique=True, index=True)

class BioMatchCount(Base):
    __tablename__="bio_match_count"

    id = Column(Integer, primary_key=True, index=True)
    count = Column(Integer, default=0)