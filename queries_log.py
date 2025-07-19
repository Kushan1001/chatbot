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
db_url = 'postgresql+psycopg2://kushan:98o4IThpeKEkpWl2zmLz8SvOIM6wV440@dpg-d1sd68fdiees73ffpp6g-a/chat_history_4gwf?sslmode=require'
engine = create_engine(db_url, echo=False)

Base.metadata.create_all(engine)
