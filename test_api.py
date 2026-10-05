#!/usr/bin/env python3
"""Быстрый тест API Topology — регистрация, логин и создание первой 3D модели."""

import sys
import time

import requests

API = "http://localhost:8000"
TIMEOUT = 10  # секунд на каждый запрос


def request(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        print(f"❌ API недоступен ({API}): {exc}")
        sys.exit(1)


# 1. Регистрация
print("📝 Регистрация...")
email = f"test_{int(time.time())}@example.com"
user_data = {"email": email, "password": "test123456"}
r = request("POST", "/auth/register", json=user_data)
if r.status_code != 201:
    print(f"❌ Ошибка регистрации: {r.text}")
    sys.exit(1)

user = r.json()
print(f"✅ Пользователь создан: {user['user']['email']}")

# 2. Логин (регистрация уже вернула токен, но проверяем и эндпоинт логина)
print("\n🔑 Логин...")
r = request("POST", "/auth/login", json=user_data)
if r.status_code != 200:
    print(f"❌ Ошибка логина: {r.text}")
    sys.exit(1)

token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print("✅ Токен получен")

# 3. Создание задачи на 3D модель
print("\n🏗️  Создание задачи на генерацию 3D модели...")
job_data = {
    "center": {
        "lon": 56.343056,  # долгота, в.д.
        "lat": 58.050485,  # широта, с.ш.
    },
    "radius_m": 500,  # 500м радиус (0.5-3 км)
    "detail": "LOD1",  # Level of Detail: LOD1 (базовая детализация)
}

r = request("POST", "/jobs", json=job_data, headers=headers)
if r.status_code != 201:
    print(f"❌ Ошибка создания задачи: {r.text}")
    sys.exit(1)

job = r.json()
job_id = job["id"]
print(f"✅ Задача создана: {job_id}")
print(f"   Статус: {job['status']}")

# 4. Проверка статуса
print("\n📊 Проверка статуса задачи...")
r = request("GET", f"/jobs/{job_id}", headers=headers)
if r.status_code != 200:
    print(f"❌ Ошибка: {r.text}")
    sys.exit(1)

job = r.json()
print(f"✅ Текущий статус: {job['status']}")
steps = sorted(job["steps"], key=lambda s: s["step_order"])
current = next((s for s in steps if s["status"] != "done"), None)
if current is not None:
    print(f"   Текущий этап: {current['step_name']} ({current['status']})")
elif steps:
    print("   Все этапы завершены")
print("\n✨ Готово! Задача обрабатывается в фоне.")
print(f"📍 ID задачи: {job_id}")
print(f"🔍 Проверяй статус: GET {API}/jobs/{job_id}")
print(f"📥 Скачивай результаты: GET {API}/models/{job_id}/files")
