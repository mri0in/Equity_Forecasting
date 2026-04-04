FROM python:3.11-slim

WORKDIR /app

# Fix import path
ENV PYTHONPATH=/app

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY . .

CMD ["uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]