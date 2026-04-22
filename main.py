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

app = FastAPI(title="VisionCity: AI-Inspector 24/7 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "256"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "60"))
QWEN_TIMEOUT_SECONDS = float(os.getenv("QWEN_TIMEOUT_SECONDS", "15"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
FAST_MODE_IMAGE_BYTES = int(os.getenv("FAST_MODE_IMAGE_BYTES", "200000"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1000"))
OLLAMA_RETRY_NUM_PREDICT = int(os.getenv("OLLAMA_RETRY_NUM_PREDICT", "3000"))
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
REAL_INCIDENT_CATEGORIES = {"мусор", "дороги", "свет", "люки", "инфраструктура"}
CATEGORY_ALIASES = {
    "мусор": "мусор",
    "trash": "мусор",
    "garbage": "мусор",
    "отходы": "мусор",
    "дорога": "дороги",
    "дороги": "дороги",
    "яма": "дороги",
    "ямы": "дороги",
    "pothole": "дороги",
    "road": "дороги",
    "свет": "свет",
    "фонарь": "свет",
    "фонари": "свет",
    "lighting": "свет",
    "люк": "люки",
    "люки": "люки",
    "manhole": "люки",
    "инфраструктура": "инфраструктура",
    "infrastructure": "инфраструктура",
}
SERVICE_BY_CATEGORY = {
    "мусор": "клининг",
    "дороги": "дорожная служба",
    "свет": "электрики",
    "люки": "аварийная служба",
    "инфраструктура": "служба эксплуатации инфраструктуры",
}


class IncidentAnalysis(BaseModel):
    is_fake: bool
    confidence: float
    problem: str
    category: str
    trash_type: str
    volume: str
    urgency: str
    recommendation: str

IMAGE_ANALYSIS_PROMPT = """
Ты — VisionCity: AI-Инспектор 24/7 для городских коммунальных служб.
Твоя задача — по фотографии определить, есть ли реальная городская проблема, классифицировать ее, оценить срочность и дать рекомендацию ответственной службе.

ГОРОДСКИЕ ПРОБЛЕМЫ, КОТОРЫЕ НУЖНО ПРИНИМАТЬ:
- мусор: свалка, переполненная урна или контейнер, разбросанные отходы, строительный мусор, покрышки.
- дороги: яма, разрушенный асфальт, провал, опасная трещина, повреждение проезжей части или тротуара.
- свет: неработающий, поврежденный или упавший фонарь, проблема с уличным освещением.
- люки: открытый, отсутствующий, поврежденный или просевший люк, опасная крышка колодца.
- инфраструктура: сломанная остановка, знак, ограждение, скамейка, бордюр, плитка, оголенный провод или другое повреждение городской инфраструктуры.

ПРАВИЛА is_fake:
1. is_fake = true, если изображение не является фото городской проблемы:
   - мем, скриншот, постер, рисунок, сгенерированная картинка;
   - селфи, еда, животные, интерьер, реклама, документ или другой нерелевантный контент;
   - обычная чистая улица без видимой проблемы.
2. is_fake = false, если на фото есть хотя бы одна городская проблема из списка выше.
3. Если фото реальное, но плохого качества, ставь is_fake = false только если проблему можно распознать. Если проблема неразличима, ставь низкий confidence и recommendation: "требуется ручная проверка".

CATEGORY:
- Верни ровно одну категорию: "мусор", "дороги", "свет", "люки", "инфраструктура".
- Если проблем несколько, выбери самую опасную.
- Не ставь категорию "мусор" автоматически: классифицируй динамически по содержанию фото.

URGENCY:
- high: опасно для жизни/транспорта/пешеходов. Примеры: открытый люк, большая яма на дороге, оголенный провод, упавший столб, опасный провал.
- medium: мешает городу, но не выглядит немедленно опасным. Примеры: мусорная куча, переполненный контейнер, сломанный фонарь, поврежденная остановка.
- low: небольшая или некритичная проблема. Примеры: мелкий мусор, небольшая трещина, легкое повреждение элемента инфраструктуры.

RECOMMENDATION:
- Всегда укажи ответственную службу и конкретное действие:
  - мусор → клининг;
  - дороги → дорожная служба;
  - свет → электрики;
  - люки → аварийная служба;
  - инфраструктура → служба эксплуатации инфраструктуры.

ПОЛЯ ОТВЕТА:
- problem: кратко опиши, что сломано/загрязнено и где это видно на фото.
- category: одна из категорий выше.
- trash_type: поле оставлено для совместимости; заполни типом проблемы, например "бытовой мусор", "яма", "открытый люк", "неработающий фонарь", "сломанное ограждение".
- volume: масштаб проблемы: "малый", "средний" или "большой".
- urgency: "low", "medium" или "high".
- recommendation: ответственная служба + действие.

ОТВЕТ СТРОГО JSON:
{
  "is_fake": boolean,
  "confidence": float,
  "problem": "краткое описание городской проблемы",
  "category": "мусор|дороги|свет|люки|инфраструктура",
  "trash_type": "тип проблемы",
  "volume": "малый|средний|большой",
  "urgency": "low|medium|high",
  "recommendation": "ответственная служба и конкретное действие"
}
""".strip()

IMAGE_ANALYSIS_RETRY_PROMPT = """
/no_think
Ты — AI-инспектор городских проблем. Проанализируй фото и верни только JSON без рассуждений, markdown и пояснений.

Прими только реальные городские проблемы: мусор, дороги, свет, люки, инфраструктура.
Если на фото нет такой проблемы, поставь is_fake=true.
Если проблема есть, поставь is_fake=false и выбери одну категорию:
"мусор", "дороги", "свет", "люки", "инфраструктура".

urgency:
- high: опасно для людей/транспорта.
- medium: заметная городская проблема без немедленной опасности.
- low: небольшая некритичная проблема.

recommendation должна назвать ответственную службу:
мусор → клининг; дороги → дорожная служба; свет → электрики; люки → аварийная служба; инфраструктура → служба эксплуатации инфраструктуры.

JSON schema:
{
  "is_fake": boolean,
  "confidence": float,
  "problem": "краткое описание",
  "category": "мусор|дороги|свет|люки|инфраструктура",
  "trash_type": "тип проблемы",
  "volume": "малый|средний|большой",
  "urgency": "low|medium|high",
  "recommendation": "служба и действие"
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
        "problem": "Обнаружена возможная городская проблема (fast mode)",
        "category": "инфраструктура",
        "trash_type": "требуется уточнение",
        "volume": "средний",
        "urgency": "medium",
        "recommendation": "Службе эксплуатации инфраструктуры требуется проверить заявку.",
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


def _normalize_category(value: Any) -> str:
    normalized = _normalized_text(value)
    return CATEGORY_ALIASES.get(normalized, normalized)


def _has_real_incident_category(analysis: dict[str, Any]) -> bool:
    return _normalize_category(analysis.get("category")) in REAL_INCIDENT_CATEGORIES


def _ensure_service_recommendation(category: str, recommendation: Any) -> str:
    service = SERVICE_BY_CATEGORY.get(category)
    current = str(recommendation or "").strip()

    if not service:
        return current or "Требуется ручная проверка и назначение ответственной службы."

    if service.lower() in current.lower():
        return current

    action = current or "требуется проверить заявку и устранить проблему."
    return f"{service}: {action}"


def _fix_llm_output(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Если модель нашла валидную городскую категорию, принимаем заявку как реальную
    и дополняем рекомендацию ответственной службой.
    """
    normalized = dict(analysis)
    category = _normalize_category(normalized.get("category"))
    
    if category in REAL_INCIDENT_CATEGORIES:
        normalized["category"] = category
        normalized["is_fake"] = False

        current_conf = float(normalized.get("confidence") or 0.0)
        if current_conf < 0.7:
            normalized["confidence"] = 0.7

        normalized["recommendation"] = _ensure_service_recommendation(
            category,
            normalized.get("recommendation"),
        )
        logger.info("analysis_auto_accepted category=%s", category)
        
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
def _build_ollama_chat_body(
    *,
    image_base64: str,
    prompt: str,
    num_predict: int,
) -> dict[str, Any]:
    return {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": num_predict,
        },
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
    }


async def analyze_with_llm(image_base64: str) -> dict[str, Any]:
    attempts = [
        ("primary", IMAGE_ANALYSIS_PROMPT, OLLAMA_NUM_PREDICT),
        (
            "json_retry",
            IMAGE_ANALYSIS_RETRY_PROMPT,
            max(OLLAMA_NUM_PREDICT, OLLAMA_RETRY_NUM_PREDICT),
        ),
    ]
    
    start = time.perf_counter()

    last_error: Exception | None = None
    for attempt_name, prompt, num_predict in attempts:
        body = _build_ollama_chat_body(
            image_base64=image_base64,
            prompt=prompt,
            num_predict=num_predict,
        )

        response = await app.state.ollama_client.post("/chat", json=body)
        response.raise_for_status()

        response_payload = response.json()
        message = response_payload.get("message", {})
        content = message.get("content", "")
        if not content.strip():
            last_error = ValueError("Empty LLM response")
            logger.warning(
                "LLM returned empty content attempt=%s done_reason=%s thinking_chars=%s message_keys=%s",
                attempt_name,
                response_payload.get("done_reason"),
                len(str(message.get("thinking", ""))),
                sorted(message.keys()),
            )
            continue

        try:
            result = _extract_json(content)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning("LLM returned invalid JSON attempt=%s error=%s", attempt_name, exc)
            continue

        logger.info(
            "LLM took %sms attempt=%s",
            int((time.perf_counter() - start) * 1000),
            attempt_name,
        )
        return result

    raise last_error or ValueError("LLM analysis failed")

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
