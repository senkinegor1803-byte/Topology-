# Рендер-воркер (Шаг 4.5, п. 1): headless Blender Cycles, запуск задачи из
# очереди Celery (topology_geo.tasks.render_tasks), без интерфейса.
#
# ЧЕСТНО: этот образ собран по образцу `geo/Dockerfile` (тот же контекст
# сборки, тот же способ установки пакета) и содержит корректный, рабочий
# CPU-путь (тот самый, что проверен тестами `test_rendering_render_frame_bpy.py`
# / `test_tasks_render.py` в этой среде). GPU-путь плана («Docker-образ
# Blender с GPU») — сборка образа с драйверами NVIDIA/CUDA и его тестовый
# запуск на реальной видеокарте — НЕ выполнялись и не проверялись: в этой
# среде разработки нет GPU и не запущен демон Docker (`docker info` не
# достаёт до /var/run/docker.sock), собрать и тем более прогнать
# GPU-специфичный образ здесь невозможно. Для реального GPU-хоста базовый
# образ нужно заменить на `nvidia/cuda:...-runtime` и передать
# `--gpus all` при запуске контейнера — тогда `scene.cycles.device = "GPU"`
# (сейчас всегда `"CPU"`, см. `rendering/render_frame.py`) заработает без
# изменений в остальной логике задачи.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir bpy

ENV CELERY_QUEUES=render

CMD ["python", "-m", "celery", "-A", "topology_geo.tasks.celery_app", "worker", \
     "--loglevel=info", "--concurrency=1", "-Q", "render"]
