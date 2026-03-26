from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./storage/incidents.db")


def _ensure_sqlite_parent_dir() -> None:
    if not DATABASE_URL.startswith("sqlite+aiosqlite:///"):
        return

    db_file = DATABASE_URL.removeprefix("sqlite+aiosqlite:///")
    db_dir = os.path.dirname(db_file)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


engine = create_async_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(AsyncAttrs, DeclarativeBase):
    pass


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    analysis_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


async def init_db() -> None:
    _ensure_sqlite_parent_dir()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def save_incident(
    *,
    lat: float,
    lng: float,
    image_path: str,
    analysis: dict[str, Any],
) -> int:
    async with SessionLocal() as session:
        record = IncidentRecord(
            lat=lat,
            lng=lng,
            image_path=image_path,
            analysis_json=json.dumps(analysis, ensure_ascii=False),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id
