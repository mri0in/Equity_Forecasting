FROM python:3.11-slim

WORKDIR /app

# Install dependencies FIRST
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy project
COPY . .

EXPOSE 8000

CMD ["python", "entrypoints/run_api.py"]