#!/usr/bin/env python3
"""Быстрый тест API Topology — регистрация, логин и создание первой 3D модели."""

import requests
import json
import time

API = "http://localhost:8000"

# 1. Регистрация
print("📝 Регистрация...")
email = f"test_{int(time.time())}@example.com"
user_data = {"email": email, "password": "test123456"}
r = requests.post(f"{API}/auth/register", json=user_data)
if r.status_code != 201:
    print(f"❌ Ошибка регистрации: {r.text}")
    exit(1)

user = r.json()
print(f"✅ Пользователь создан: {user['user']['email']}")

# 2. Логин
print("\n🔑 Логин...")
r = requests.post(f"{API}/auth/login", json=user_data)
if r.status_code != 200:
    print(f"❌ Ошибка логина: {r.text}")
    exit(1)

token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}
print(f"✅ Токен получен")

# 3. Создание задачи на 3D модель
print("\n🏗️  Создание задачи на генерацию 3D модели...")
job_data = {
    "name": "Perm Circus Area",
    "center_lon": 58.050485,  # Пермь, цирк (долгота)
    "center_lat": 56.343056,  # Пермь, цирк (широта)
    "radius_m": 500,          # 500м радиус
    "project": "perm-project"
}

r = requests.post(f"{API}/jobs", json=job_data, headers=headers)
if r.status_code != 201:
    print(f"❌ Ошибка создания задачи: {r.text}")
    exit(1)

job = r.json()
job_id = job["id"]
print(f"✅ Задача создана: {job_id}")
print(f"   Статус: {job['status']}")
print(f"   Этап: {job['current_step']}")

# 4. Проверка статуса
print("\n📊 Проверка статуса задачи...")
r = requests.get(f"{API}/jobs/{job_id}", headers=headers)
if r.status_code == 200:
    job = r.json()
    print(f"✅ Текущий статус: {job['status']}")
    print(f"   Этап: {job['current_step']}")
    print(f"\nГотово! Задача обрабатывается в фоне.")
    print(f"Проверяй статус: GET /jobs/{job_id}")
    print(f"Скачивай результаты: GET /models/{job_id}/files")
else:
    print(f"❌ Ошибка: {r.text}")
