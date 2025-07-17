import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_query = Column(Text)

# Use PostgreSQL if available, fallback to local SQLite
db_url = os.environ.get("DATABASE_URL", "sqlite:///chat_history.db")
engine = create_engine(db_url, echo=False)
