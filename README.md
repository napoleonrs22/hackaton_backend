# VisionCity: AI-Inspector 24/7 Backend

Бэкенд-сервис на `FastAPI` для приёма фотографий городских проблем, их AI-анализа, сохранения инцидентов и базового поиска похожих кейсов.

Сервис принимает изображение и координаты, пытается определить, есть ли на фото реальная городская проблема, относит её к одной из категорий и возвращает результат в одном из статусов:

- `success` — проблема подтверждена
- `needs_review` — фото похоже на реальное, но уверенности модели недостаточно
- `rejected` — изображение похоже на фейк или нерелевантный контент

## Что умеет сервис

- принимает изображение через `multipart/form-data`
- нормализует и сжимает фото перед анализом
- анализирует изображение через `Ollama` и мультимодальную модель `qwen3-vl:4b`
- автоматически сохраняет инциденты в БД
- хранит изображения на диске
- ведёт векторную память через `ChromaDB` для переиспользования похожих кейсов
- использует in-memory кэш для повторного анализа одинаковых изображений
- поддерживает быстрый fallback-режим для крупных файлов

## Категории инцидентов

Сервис работает со следующими типами городских проблем:

- `мусор`
- `дороги`
- `свет`
- `люки`
- `инфраструктура`

Для каждой категории сервис старается добавить рекомендацию ответственной службе:

- `мусор` → `клининг`
- `дороги` → `дорожная служба`
- `свет` → `электрики`
- `люки` → `аварийная служба`
- `инфраструктура` → `служба эксплуатации инфраструктуры`

## Архитектура

Основные компоненты:

- `main.py` — FastAPI-приложение, HTTP API, пайплайн анализа изображений
- `services/db.py` — работа с БД через `SQLAlchemy Async`
- `services/vector_db.py` — работа с `ChromaDB`
- `storage/uploads` — сохранённые изображения
- `storage/chroma` — файлы векторного индекса

Поток обработки запроса:

1. Клиент загружает изображение и координаты.
2. Сервис валидирует данные и оптимизирует изображение.
3. Если такой файл уже анализировался, ответ берётся из кэша.
4. Для больших изображений включается `fast mode`.
5. В обычном режиме изображение отправляется в `Ollama`.
6. Результат нормализуется и сохраняется в БД.
7. Успешные кейсы дополнительно индексируются в `ChromaDB`.

## Технологии

- Python 3.11+
- FastAPI
- Uvicorn
- SQLAlchemy Async
- SQLite или PostgreSQL
- ChromaDB
- Ollama
- Pillow
- httpx

## Требования

Для локального запуска нужны:

- `Python 3.11+`
- установленный `Ollama`
- загруженная модель `qwen3-vl:4b` или другая совместимая мультимодальная модель

Если вы используете `docker-compose`, отдельно нужен доступный хостовый `Ollama`, потому что контейнер backend по умолчанию обращается к:

```text
http://host.docker.internal:11434
```

## Быстрый старт

### 1. Установить зависимости

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Поднять Ollama и модель

Пример:

```bash
ollama pull qwen3-vl:4b
ollama serve
```

По умолчанию приложение ждёт `Ollama` на `http://localhost:11434`.

### 3. Запустить бэкенд

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

После запуска:

- Swagger UI: `http://localhost:8010/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Запуск через Docker Compose

В проекте есть готовый `docker-compose.yml`.

```bash
docker compose up --build
```

По умолчанию будут подняты:

- `db` — PostgreSQL 15
- `backend` — FastAPI backend

Порты:

- backend: `http://localhost:8010`
- postgres: `localhost:5432`

Важно:

- в compose backend использует `DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/hackathon`
- `Ollama` не запускается в compose и должен быть доступен с хоста
- данные Postgres и загруженные файлы сохраняются в Docker volumes

## Конфигурация

Ниже перечислены основные переменные окружения.

### AI и обработка изображений

| Переменная | По умолчанию | Описание |
|---|---:|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Базовый URL Ollama |
| `OLLAMA_MODEL` | `qwen3-vl:4b` | Модель для анализа изображений |
| `QWEN_TIMEOUT_SECONDS` | `15` | Таймаут чтения ответа от модели |
| `OLLAMA_KEEP_ALIVE` | `15m` | Keep-alive для модели в Ollama |
| `OLLAMA_NUM_CTX` | `2048` | Размер контекста |
| `OLLAMA_NUM_PREDICT` | `1000` | Лимит генерации для основного запроса |
| `OLLAMA_RETRY_NUM_PREDICT` | `3000` | Лимит генерации для retry-запроса |
| `LLM_MAX_CONCURRENCY` | `1` | Максимум одновременных обращений к LLM |
| `IMAGE_MAX_SIZE` | `256` | Максимальная сторона изображения после ресайза |
| `IMAGE_JPEG_QUALITY` | `60` | Качество JPEG после оптимизации |
| `FAST_MODE_IMAGE_BYTES` | `200000` | Порог включения fast mode |

### Качество классификации

| Переменная | По умолчанию | Описание |
|---|---:|---|
| `NEEDS_REVIEW_CONFIDENCE` | `0.5` | Ниже этого значения кейс уходит на ручную проверку |
| `REJECT_FAKE_CONFIDENCE` | `0.95` | Порог уверенного отклонения фейков |
| `VECTOR_REUSE_SCORE` | `0.9` | Порог переиспользования похожего кейса из Chroma |

### Хранилища

| Переменная | По умолчанию | Описание |
|---|---:|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./storage/incidents.db` | Строка подключения к БД |
| `UPLOAD_DIR` | `storage` | Корневая директория хранения |
| `CHROMA_PATH` | `storage/chroma` | Путь к persistent-хранилищу Chroma |
| `CHROMA_COLLECTION` | `incidents` | Имя коллекции в Chroma |

### Кэш анализа

| Переменная | По умолчанию | Описание |
|---|---:|---|
| `IMAGE_ANALYSIS_CACHE_SIZE` | `256` | Размер in-memory кэша |
| `IMAGE_ANALYSIS_CACHE_TTL_SECONDS` | `900` | Время жизни кэша в секундах |

### Значения из `docker-compose.yml`

В `docker-compose.yml` часть настроек уже переопределена, например:

- `IMAGE_MAX_SIZE=620`
- `IMAGE_JPEG_QUALITY=85`
- `QWEN_TIMEOUT_SECONDS=120`
- `FAST_MODE_IMAGE_BYTES=800000`
- `UPLOAD_DIR=/app/storage`
- `CHROMA_PATH=/app/storage/chroma`

## API

### `POST /analyze-image`

Анализирует изображение и создаёт инцидент.

Формат запроса: `multipart/form-data`

Поля:

- `file` — изображение
- `lat` — широта
- `lng` — долгота

Пример:

```bash
curl -X POST "http://localhost:8000/analyze-image" \
  -F "file=@test.jpg" \
  -F "lat=43.238949" \
  -F "lng=76.889709"
```

Пример ответа при успехе:

```json
{
  "status": "success",
  "incident_id": 12,
  "analysis": {
    "is_fake": false,
    "confidence": 0.87,
    "problem": "Открытый люк у края проезжей части",
    "category": "люки",
    "trash_type": "открытый люк",
    "volume": "средний",
    "urgency": "high",
    "recommendation": "аварийная служба: оградить участок и закрыть люк"
  },
  "fast_mode": false,
  "cache_hit": false
}
```

Возможные статусы:

- `success`
- `needs_review`
- `rejected`

Особенности:

- даже при `needs_review` или `rejected` запись может быть сохранена в БД
- при сбое LLM сервис возвращает безопасный fallback-результат с рекомендацией ручной проверки
- если изображение слишком большое, может сработать `fast_mode`

### `GET /incidents`

Возвращает список сохранённых инцидентов в обратном хронологическом порядке.

Пример:

```bash
curl "http://localhost:8000/incidents"
```

### `DELETE /incidents/{incident_id}`

Удаляет инцидент из БД, векторного индекса и с диска.

Пример:

```bash
curl -X DELETE "http://localhost:8000/incidents/12"
```

## Формат анализа

Поле `analysis` обычно содержит:

```json
{
  "is_fake": false,
  "confidence": 0.0,
  "problem": "Текстовое описание проблемы",
  "category": "мусор",
  "trash_type": "бытовой мусор",
  "volume": "малый",
  "urgency": "low",
  "recommendation": "клининг: убрать мусор и проверить контейнерную площадку"
}
```

Пояснения:

- `is_fake` — считает ли модель изображение нерелевантным
- `confidence` — уверенность модели
- `problem` — краткое описание проблемы
- `category` — одна из бизнес-категорий
- `trash_type` — конкретный подтип проблемы
- `volume` — `малый`, `средний`, `большой`
- `urgency` — `low`, `medium`, `high`
- `recommendation` — ответственная служба и действие

## Хранение данных

По умолчанию без переменной `DATABASE_URL` сервис работает на SQLite:

```text
storage/incidents.db
```

Также создаются:

- `storage/uploads` — изображения
- `storage/chroma` — индекс похожих кейсов

В Docker-сценарии обычно используется PostgreSQL, а файловое хранилище монтируется в volume.

## Ограничения и особенности

- CORS открыт для всех источников
- аутентификация и авторизация не реализованы
- API не проверяет бизнес-доступ пользователей
- качество результата сильно зависит от модели в `Ollama`
- `ChromaDB` используется как вспомогательная память, а не как источник истины
- векторная индексация делается только для успешных кейсов без `fast_mode` и без cache hit

## Полезно при отладке

Если сервис не отвечает как ожидается, проверьте:

1. доступен ли `Ollama` по `OLLAMA_BASE_URL`
2. загружена ли модель `OLLAMA_MODEL`
3. корректен ли `DATABASE_URL`
4. есть ли права на запись в `UPLOAD_DIR`
5. не истёк ли таймаут `QWEN_TIMEOUT_SECONDS`

## Структура проекта

```text
.
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
└── services
    ├── db.py
    └── vector_db.py
```
