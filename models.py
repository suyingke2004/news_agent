from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass


class VerificationSession(Base):
    __tablename__ = "verification_session"

    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    start_time = Column(DateTime, default=datetime.utcnow)
    title = Column(String(200), nullable=True)
    overall_verdict = Column(String(20), nullable=True)
    overall_confidence = Column(String(10), nullable=True)

    messages = relationship(
        "VerificationMessage",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<VerificationSession(id='{self.id}', verdict='{self.overall_verdict}')>"


class VerificationMessage(Base):
    __tablename__ = "verification_message"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey("verification_session.id"), nullable=False)
    message_type = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("VerificationSession", back_populates="messages")

    def __repr__(self):
        return (
            f"<VerificationMessage(session_id='{self.session_id}', "
            f"type='{self.message_type}')>"
        )
