from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from datetime import datetime, timezone

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user_query = Column(Text)


engine = create_engine("sqlite:///chat_history.db")
Base.metadata.create_all(engine)
print('Database created successfully !')

