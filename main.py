from __future__ import annotations

import base64
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from starlette.concurrency import run_in_threadpool

from services.db import init_db, save_incident
from services.vector_db import add_incident, build_incident_memory_text, warmup_vector_db

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
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "640"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))
QWEN_TIMEOUT_SECONDS = float(os.getenv("QWEN_TIMEOUT_SECONDS", "60"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
UPLOAD_ROOT = Path(os.getenv("UPLOAD_DIR", "storage"))
UPLOADS_DIR = UPLOAD_ROOT / "uploads"

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
Ты AI-эколог Smart City.
Твоя задача — анализировать изображения мусора и защищать систему от фейковых заявок.

ПРАВИЛА:
1. Определи, является ли изображение реальным:
   - мемы, скриншоты, селфи, картинки из интернета → is_fake = true
   - реальные фото улицы, двора, дороги → is_fake = false
2. Если is_fake = false:
   - category: мусор | дороги | свет
   - trash_type: бытовой | строительный | пластик | смешанный | нет
   - volume: маленький | средний | большой | нет
   - urgency: low | medium | high
   - дай рекомендацию
3. Если не уверен или фото размыто:
   - is_fake = true
   - confidence < 0.5

ОТВЕТ СТРОГО JSON:
{
  "is_fake": boolean,
  "confidence": float,
  "problem": string,
  "category": string,
  "trash_type": string,
  "volume": string,
  "urgency": string,
  "recommendation": string
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

# ==================== LLM ====================
async def analyze_with_llm(image_base64: str) -> dict[str, Any]:
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
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

# ==================== STARTUP ====================
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

# ==================== ENDPOINT ====================
@app.post("/analyze-image")
async def analyze_image(
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
    image_base64 = base64.b64encode(img).decode()

    analysis = None
    llm_failed = False
    try:
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

    # 🚨 АНТИ-ФЕЙК
    is_rejected = not llm_failed and (analysis["is_fake"] or analysis["confidence"] < 0.5)
    # 💾 DB (Сохраняем ВСЕГДА, чтобы собирать статистику и банить спамеров)
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

    if llm_failed:
        logger.info("TOTAL %sms (NEEDS_REVIEW)", int((time.perf_counter() - start) * 1000))
        return {
            "status": "needs_review",
            "incident_id": incident_id,
            "message": "LLM недоступен или не завершил анализ. Требуется ручная проверка.",
            "analysis": analysis,
        }

    # Прерываем флоу, если это фейк
    if is_rejected:
        logger.info("TOTAL %sms (REJECTED)", int((time.perf_counter() - start) * 1000))
        return {
            "status": "rejected",
            "incident_id": incident_id,
            "message": "Изображение отклонено системой модерации.",
            "analysis": analysis
        }

    # 🧠 VECTOR DB (Только для реальных инцидентов)
    try:
        await run_in_threadpool(
            add_incident,
            build_incident_memory_text(analysis),
            {
                "lat": lat_f,
                "lng": lng_f,
                "category": analysis["category"],
                "urgency": analysis["urgency"],
            },
            str(incident_id),
        )
    except Exception:
        logger.exception("vector_db_failed")

    logger.info("TOTAL %sms (SUCCESS)", int((time.perf_counter() - start) * 1000))

    return {
        "status": "success",
        "incident_id": incident_id,
        "analysis": analysis
    }
