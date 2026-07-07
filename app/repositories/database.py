"""
Database engine + ORM models.

SQLite for local dev (DATABASE_URL=sqlite+aiosqlite:///./data/app.db).
Swapping to Postgres later = change DATABASE_URL to a postgresql+asyncpg://
URL — the repositories below use only standard SQLAlchemy Core/ORM calls,
nothing SQLite-specific, so no repository code needs to change.
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config.settings import get_settings


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(16))
    upload_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()

        if settings.database_url.startswith("sqlite"):
            db_path = settings.database_url.split("///")[-1] if "///" in settings.database_url else "./data/app.db"
            db_file = Path(db_path)
            if db_file.parent != Path(""):
                db_file.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncSession:
    """FastAPI dependency — yields a session, closes it after the request."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session
