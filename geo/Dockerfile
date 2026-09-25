FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY docker/entrypoint.py ./entrypoint.py

CMD ["python", "entrypoint.py"]
