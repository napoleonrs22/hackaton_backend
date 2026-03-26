FROM python:3.11-slim

# Запрещаем Python писать .pyc файлы и буферизовать stdout/stderr (полезно для логов в Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Устанавливаем только самое необходимое для сборки ChromaDB (hnswlib) и asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Обновляем pip и ставим зависимости без кэша для уменьшения веса образа
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]