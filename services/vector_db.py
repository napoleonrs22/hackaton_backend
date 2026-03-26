from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "incidents")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.getenv("UPLOAD_DIR", "storage"), "chroma"))

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def _prepare_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}

    prepared: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            prepared[key] = value
        else:
            prepared[key] = json.dumps(value, ensure_ascii=False)
    return prepared


@lru_cache(maxsize=1)
def get_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def warmup_vector_db() -> None:
    get_collection()


def build_incident_memory_text(analysis: dict[str, Any]) -> str:
    problem = str(analysis.get("problem", "")).strip()
    recommendation = str(analysis.get("recommendation", "")).strip()
    return "\n".join(part for part in [problem, recommendation] if part)


def add_incident(
    text: str,
    metadata: dict[str, Any] | None = None,
    incident_id: str | None = None,
) -> str:
    if not text.strip():
        return ""

    collection = get_collection()
    document_id = incident_id or hashlib.sha1(text.encode("utf-8")).hexdigest()
    collection.upsert(
        ids=[document_id],
        documents=[text],
        metadatas=[_prepare_metadata(metadata)],
    )
    return document_id


def search_similar(text: str, n_results: int = 3) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    results = collection.query(
        query_texts=[text],
        n_results=min(n_results, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    similar_cases: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = round(max(0.0, 1.0 - float(distance)), 4) if distance is not None else None
        similar_cases.append(
            {
                "text": document,
                "metadata": metadata or {},
                "score": score,
            }
        )

    return similar_cases
