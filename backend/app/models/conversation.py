import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), default="New Chat", nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")  # type: ignore # noqa: F821
    messages: Mapped[List["Message"]] = relationship(  # type: ignore # noqa: F821
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.message_order",
    )

    @property
    def message_count(self) -> int:
        if hasattr(self, "_message_count"):
            return self._message_count
        if "messages" in self.__dict__ and self.messages is not None:
            return len(self.messages)
        return 0

    @message_count.setter
    def message_count(self, value: int) -> None:
        self._message_count = value

