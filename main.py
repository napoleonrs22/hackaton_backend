from __future__ import annotations

from collections import OrderedDict
from asyncio import Semaphore
import base64
import hashlib
import json
import logging
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from services.db import delete_incident as delete_incident_record
from services.db import init_db, list_incidents, save_incident
from services.vector_db import (
    add_incident,
    build_incident_memory_text,
    delete_incident as delete_vector_incident,
    search_similar,
    warmup_vector_db,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart City AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "620"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))
QWEN_TIMEOUT_SECONDS = float(os.getenv("QWEN_TIMEOUT_SECONDS", "60"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
FAST_MODE_IMAGE_BYTES = int(os.getenv("FAST_MODE_IMAGE_BYTES", "800000"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "120"))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "1")))
VECTOR_REUSE_SCORE = float(os.getenv("VECTOR_REUSE_SCORE", "0.9"))
IMAGE_ANALYSIS_CACHE_SIZE = max(1, int(os.getenv("IMAGE_ANALYSIS_CACHE_SIZE", "256")))
IMAGE_ANALYSIS_CACHE_TTL_SECONDS = max(1, int(os.getenv("IMAGE_ANALYSIS_CACHE_TTL_SECONDS", "900")))
NEEDS_REVIEW_CONFIDENCE = float(os.getenv("NEEDS_REVIEW_CONFIDENCE", "0.5"))
REJECT_FAKE_CONFIDENCE = float(os.getenv("REJECT_FAKE_CONFIDENCE", "0.95"))
UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", "storage"))
UPLOADS_DIR = UPLOAD_ROOT / "uploads"
LLM_SEMAPHORE = Semaphore(LLM_MAX_CONCURRENCY)
IMAGE_ANALYSIS_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
REAL_INCIDENT_CATEGORIES = {"мусор", "дороги", "свет"}


# ==================== SCHEMA ====================
class IncidentAnalysis(BaseModel):
    is_fake: bool
    confidence: float
    problem: str
    category: str
    trash_type: str
    volume: str
    urgency: str
    recommendation: str

# ==================== PROMPT ====================
IMAGE_ANALYSIS_PROMPT = """
Ты — узкоспециализированный AI-инспектор Smart City по контролю чистоты.
Твоя единственная цель: выявлять незаконные свалки, переполненные контейнеры и разбросанный мусор.

ПРАВИЛА КЛАССИФИКАЦИИ:
1. is_fake = true, если:
   - Это НЕ фото (мем, скриншот, нейросеть, постер).
   - На фото НЕТ мусора (просто чистая улица, парк, люди, машины, ямы на дорогах или сломанные фонари). Для этой системы отсутствие мусора = нецелевая заявка (фейк).
2. is_fake = false, если:
   - На фото четко виден мусор (любого типа).
   - Фото реальное, но плохого качества (размыто, темно) — в этом случае ставь низкий confidence и recommendation: "требуется уточнение".

ДЕТАЛИЗАЦИЯ (только если is_fake = false):
- category: всегда "мусор" (если это не мусор, ставь is_fake=true).
- trash_type: бытовой | строительный | покрышки | крупногабаритный | пластик | смешанный.
- volume: очень маленький (окурок/бутылка) | средний (пакет/бачок) | большой (свалка/гора).
- urgency: low (единичный мусор) | medium (заполненный бак) | high (стихийная свалка, токсичные отходы).

ОТВЕТ СТРОГО JSON:
{
  "is_fake": boolean,
  "confidence": float,
  "problem": "краткое описание что именно и где лежит",
  "category": "мусор",
  "trash_type": "тип",
  "volume": "объем",
  "urgency": "low|medium|high",
  "recommendation": "что сделать службе клининга"
}
""".strip()

# ==================== UTILS ====================
def _ollama_api_base(base_url: str) -> str:
    return base_url if base_url.endswith("/api") else f"{base_url}/api"

def _extract_json(content: str) -> dict[str, Any]:
    if not content.strip():
        raise ValueError("Empty LLM response")
    
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    payload = match.group(1) if match else content

    parsed = json.loads(payload)
    return IncidentAnalysis.model_validate(parsed).model_dump()

def _default_analysis() -> dict[str, Any]:
    return {
        "is_fake": False,
        "confidence": 0.0,
        "problem": "Требуется ручная проверка",
        "category": "unknown",
        "trash_type": "нет",
        "volume": "нет",
        "urgency": "low",
        "recommendation": "ИИ не смог завершить анализ. Направьте оператора на ручную проверку.",
    }


def _fast_mode_analysis() -> dict[str, Any]:
    return {
        "is_fake": False,
        "confidence": 0.6,
        "problem": "Обнаружен возможный мусор (fast mode)",
        "category": "мусор",
        "trash_type": "смешанный",
        "volume": "средний",
        "urgency": "medium",
        "recommendation": "Требуется проверка",
    }


def _ensure_storage():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def _prepare_image(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
        image.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE))
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        optimized = buffer.getvalue()

        path = UPLOADS_DIR / f"{uuid4().hex}.jpg"
        path.write_bytes(optimized)

        return str(path), optimized

    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image")


def _delete_image_file(image_path: str) -> None:
    if not image_path:
        return

    Path(image_path).unlink(missing_ok=True)


def _get_cached_analysis(image_hash: str) -> dict[str, Any] | None:
    cached_item = IMAGE_ANALYSIS_CACHE.get(image_hash)
    if cached_item is None:
        return None

    created_at, analysis = cached_item
    if time.monotonic() - created_at > IMAGE_ANALYSIS_CACHE_TTL_SECONDS:
        IMAGE_ANALYSIS_CACHE.pop(image_hash, None)
        return None

    IMAGE_ANALYSIS_CACHE.move_to_end(image_hash)
    return dict(analysis)


def _store_cached_analysis(image_hash: str, analysis: dict[str, Any]) -> None:
    IMAGE_ANALYSIS_CACHE[image_hash] = (time.monotonic(), dict(analysis))
    IMAGE_ANALYSIS_CACHE.move_to_end(image_hash)
    while len(IMAGE_ANALYSIS_CACHE) > IMAGE_ANALYSIS_CACHE_SIZE:
        IMAGE_ANALYSIS_CACHE.popitem(last=False)


def _add_incident_to_vector_store(
    analysis: dict[str, Any],
    *,
    incident_id: int,
    lat: float,
    lng: float,
    image_hash: str,
) -> None:
    try:
        add_incident(
            build_incident_memory_text(analysis),
            {
                "lat": lat,
                "lng": lng,
                "category": analysis["category"],
                "urgency": analysis["urgency"],
                "image_hash": image_hash,
                "analysis_json": analysis,
            },
            str(incident_id),
        )
    except Exception:
        logger.exception("vector_db_failed")


def _restore_analysis_from_similar(similar_cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not similar_cases:
        return None

    best_match = similar_cases[0]
    score = float(best_match.get("score") or 0.0)
    if score < VECTOR_REUSE_SCORE:
        return None

    metadata = best_match.get("metadata") or {}
    raw_analysis = metadata.get("analysis_json")
    if not raw_analysis:
        return None

    try:
        parsed = json.loads(raw_analysis) if isinstance(raw_analysis, str) else raw_analysis
        result = IncidentAnalysis.model_validate(parsed).model_dump()
    except (TypeError, json.JSONDecodeError, ValidationError, ValueError):
        logger.warning("vector_reuse_invalid_payload")
        return None

    logger.info("vector_reuse_hit score=%.4f", score)
    return result


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_real_incident_category(analysis: dict[str, Any]) -> bool:
    return _normalized_text(analysis.get("category")) in REAL_INCIDENT_CATEGORIES


def _fix_llm_output(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Если модель нашла мусор, но ошибочно считает это фейком — 
    исправляем на 'не фейк' и повышаем уверенность.
    """
    normalized = dict(analysis)
    
    # Проверяем, относится ли категория к реальным (мусор, дороги, свет)
    if _has_real_incident_category(normalized):
        # Принудительно ставим, что это НЕ фейк
        normalized["is_fake"] = False
        

        current_conf = float(normalized.get("confidence") or 0.0)
        if current_conf < 0.7:
            normalized["confidence"] = 0.7
            
        normalized["recommendation"] = "Обнаружена реальная проблема. Заявка принята автоматически."
        logger.info("analysis_auto_accepted_as_real_trash")
        
    return normalized

def _is_obvious_fake(analysis: dict[str, Any]) -> bool:
    confidence = float(analysis.get("confidence") or 0.0)
    return (
        bool(analysis.get("is_fake"))
        and confidence >= REJECT_FAKE_CONFIDENCE
        and not _has_real_incident_category(analysis)
    )


def _needs_manual_review(analysis: dict[str, Any]) -> bool:
    """
    Определяет, нужна ли ручная проверка.
    """
 
    if _has_real_incident_category(analysis) and not analysis.get("is_fake"):
        return False

    confidence = float(analysis.get("confidence") or 0.0)
    
    if confidence < NEEDS_REVIEW_CONFIDENCE:
        return True

    if bool(analysis.get("is_fake")) and not _is_obvious_fake(analysis):
        return True

    return False
async def analyze_with_llm(image_base64: str) -> dict[str, Any]:
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": [
            {
                "role": "user",
                "content": IMAGE_ANALYSIS_PROMPT,
                "images": [image_base64],
            }
        ],
    }
    
    start = time.perf_counter()

    response = await app.state.ollama_client.post("/chat", json=body)
    response.raise_for_status()

    content = response.json().get("message", {}).get("content", "")
    result = _extract_json(content)

    logger.info("LLM took %sms", int((time.perf_counter() - start) * 1000))

    return result

@app.on_event("startup")
async def startup():
    _ensure_storage()
    await init_db()
    app.state.ollama_client = httpx.AsyncClient(
        base_url=_ollama_api_base(OLLAMA_BASE_URL),
        timeout=httpx.Timeout(
            connect=5.0,
            read=QWEN_TIMEOUT_SECONDS,
            write=30.0,
            pool=5.0,
        ),
    )
    try:
        await run_in_threadpool(warmup_vector_db)
    except Exception:
        logger.exception("vector_db_warmup_failed")

@app.on_event("shutdown")
async def shutdown():
    await app.state.ollama_client.aclose()

@app.post("/analyze-image")
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lat: str = Form(...),
    lng: str = Form(...)
):
    start = time.perf_counter()
    
    try:
        lat_f, lng_f = float(lat), float(lng)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    path, img = await run_in_threadpool(_prepare_image, image_bytes)
    image_hash = hashlib.sha256(img).hexdigest()

    analysis = None
    llm_failed = False
    fast_mode_used = len(image_bytes) > FAST_MODE_IMAGE_BYTES
    cache_hit = False
    cached_analysis = _get_cached_analysis(image_hash)
    if cached_analysis is not None:
        analysis = cached_analysis
        cache_hit = True
        logger.info("analysis_cache_hit hash=%s", image_hash[:12])
    elif fast_mode_used:
        logger.info("FAST MODE enabled size=%s", len(image_bytes))
        analysis = _fast_mode_analysis()
        try:
            similar = await run_in_threadpool(
                search_similar,
                build_incident_memory_text(analysis),
                1,
            )
            analysis = _restore_analysis_from_similar(similar) or analysis
        except Exception:
            logger.exception("vector_fast_lookup_failed")
    else:
        try:
            image_base64 = base64.b64encode(img).decode()
            async with LLM_SEMAPHORE:
                analysis = await analyze_with_llm(image_base64)
        except httpx.TimeoutException:
            llm_failed = True
            logger.warning("LLM timed out after %ss", QWEN_TIMEOUT_SECONDS)
        except httpx.HTTPStatusError as exc:
            llm_failed = True
            error_body = exc.response.text[:500] if exc.response is not None else ""
            logger.error(
                "LLM returned HTTP %s body=%r",
                exc.response.status_code if exc.response is not None else "unknown",
                error_body,
            )
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError):
            llm_failed = True
            logger.exception("LLM analysis failed")
        except Exception:
            llm_failed = True
            logger.exception("Unexpected LLM error")

    if not analysis:
        analysis = _default_analysis()
    else:
        analysis = _fix_llm_output(analysis)

    if not llm_failed:
        _store_cached_analysis(image_hash, analysis)

    is_rejected = not llm_failed and _is_obvious_fake(analysis)
    needs_review = llm_failed or _needs_manual_review(analysis)

    try:
        incident_id = await save_incident(
            lat=lat_f,
            lng=lng_f,
            image_path=path,
            analysis=analysis,
        )
    except Exception:
        logger.exception("DB save failed")
        incident_id = None

    if needs_review and not is_rejected:
        logger.info("TOTAL %sms (NEEDS_REVIEW)", int((time.perf_counter() - start) * 1000))
        return {
            "status": "needs_review",
            "incident_id": incident_id,
            "message": "Требуется ручная проверка: фото выглядит реальным, но система недостаточно уверена в результате.",
            "analysis": analysis,
        }

    if is_rejected:
        logger.info("TOTAL %sms (REJECTED)", int((time.perf_counter() - start) * 1000))
        return {
            "status": "rejected",
            "incident_id": incident_id,
            "message": "Изображение похоже на явный фейк или нерелевантный контент.",
            "analysis": analysis
        }

    if incident_id is not None and not fast_mode_used and not cache_hit:
        background_tasks.add_task(
            _add_incident_to_vector_store,
            analysis,
            incident_id=incident_id,
            lat=lat_f,
            lng=lng_f,
            image_hash=image_hash,
        )

    logger.info("TOTAL %sms (SUCCESS)", int((time.perf_counter() - start) * 1000))

    return {
        "status": "success",
        "incident_id": incident_id,
        "analysis": analysis,
        "fast_mode": fast_mode_used,
        "cache_hit": cache_hit,
    }


@app.get("/incidents")
async def get_incidents():
    return await list_incidents()


@app.delete("/incidents/{incident_id}")
async def delete_incident(incident_id: int):
    deleted_record = await delete_incident_record(incident_id)
    if deleted_record is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        await run_in_threadpool(delete_vector_incident, str(incident_id))
    except Exception:
        logger.exception("vector_db_delete_failed")

    try:
        await run_in_threadpool(_delete_image_file, deleted_record["image_path"])
    except Exception:
        logger.exception("image_delete_failed")

    return {
        "status": "success",
        "incident_id": incident_id,
        "message": "Incident deleted",
    }
