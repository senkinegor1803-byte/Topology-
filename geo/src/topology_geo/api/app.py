"""FastAPI-приложение конвейера (Шаг 1.3, п. 1).

`POST /jobs` ставит задачу в очередь (Celery+Redis, Шаг 1.3, п. 2); реальное
исполнение шагов — `topology_geo.jobs.pipeline` + `topology_geo.jobs.steps`
(Шаги 1.1/1.2). Валидация входа — на уровне Pydantic-схем (`schemas.py`), так
ошибка возвращается сразу, до постановки в очередь.
"""

from __future__ import annotations

import mimetypes
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from topology_geo.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyOut,
    Center,
    ClosedContourResponse,
    ClosedContourZoneOut,
    ExchangeFolderExportResponse,
    FileOut,
    FilesResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobOut,
    JobStepOut,
    LoginRequest,
    NotificationOut,
    ProjectOut,
    RegisterRequest,
    RoleUpdateRequest,
    ShareLinkResponse,
    TokenResponse,
    UserOut,
    WebhookOut,
    WebhookRegisterRequest,
)
from topology_geo.auth import integrations as auth_integrations
from topology_geo.auth import store as auth_store
from topology_geo.constraints import store as constraints_store
from topology_geo.coords import msk59_to_wgs84, pick_msk59_zone, wgs84_to_msk59
from topology_geo.devcheck import load_environment_config
from topology_geo.export.exchange_folder import export_job_to_exchange_folder
from topology_geo.jobs import store
from topology_geo.jobs.steps import DEFAULT_STEP_NAMES
from topology_geo.tasks.pipeline_tasks import enqueue_job, get_storage

VIEWER_DIR = Path(__file__).resolve().parents[1] / "web" / "viewer"
CITYMAP_DIR = Path(__file__).resolve().parents[1] / "web" / "citymap"
PANORAMA_DIR = Path(__file__).resolve().parents[1] / "web" / "panorama"


def _connect() -> psycopg.Connection:
    config = load_environment_config()
    return psycopg.connect(config.postgres.dsn, autocommit=True)


def get_connection():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = _connect()
    try:
        auth_integrations.ensure_schema(conn)  # каскадом: jobs (1.3) -> users/... (4.10) -> api_keys/webhooks (4.11)
    finally:
        conn.close()
    yield


app = FastAPI(title="Топология: API конвейера", version="0.1.0", lifespan=lifespan)

# Веб-интерфейс
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

@app.get("/")
async def get_dashboard():
    """Главная страница — веб-интерфейс"""
    return RedirectResponse(url="/index.html", status_code=307)


@app.get("/index.html", response_class=Response)
async def get_index_html():
    """Веб-интерфейс (встроенный HTML)"""
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topology — 3D Urban Modeling</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1419; color: #e1e8ed; }

        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        header { padding: 30px 0; border-bottom: 1px solid #38444d; margin-bottom: 30px; }
        h1 { font-size: 28px; font-weight: 600; }

        .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; margin-bottom: 30px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }

        .panel { background: #192734; border: 1px solid #38444d; border-radius: 12px; padding: 20px; }

        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #b0bac0; }
        input, select { width: 100%; padding: 10px; background: #253341; border: 1px solid #38444d; color: #e1e8ed; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #1da1f2; }

        button { width: 100%; padding: 12px; background: #1da1f2; color: white; border: none; border-radius: 20px; font-weight: 600; cursor: pointer; margin-top: 10px; }
        button:hover { background: #1a91da; }
        button:disabled { background: #38444d; cursor: not-allowed; }

        .jobs-list { max-height: 600px; overflow-y: auto; }
        .job-item { background: #253341; border: 1px solid #38444d; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
        .job-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .job-title { font-weight: 600; font-size: 15px; }
        .job-status { font-size: 12px; padding: 4px 8px; border-radius: 4px; }
        .status-pending { background: #fff8e1; color: #8b6914; }
        .status-completed { background: #c3f0ca; color: #1d3a1d; }
        .status-failed { background: #ffcccc; color: #991111; }

        .progress-bar { width: 100%; height: 6px; background: #38444d; border-radius: 3px; margin: 8px 0; overflow: hidden; }
        .progress-fill { height: 100%; background: #1da1f2; border-radius: 3px; transition: width 0.3s; }

        .viewer { width: 100%; height: 600px; background: #253341; border-radius: 8px; border: 1px solid #38444d; position: relative; }
        canvas { width: 100%; height: 100%; display: block; }
        .viewer-placeholder { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; color: #657786; }

        .download-link { color: #1da1f2; text-decoration: none; font-size: 13px; }
        .download-link:hover { text-decoration: underline; }

        .step { font-size: 12px; color: #657786; margin: 3px 0; }
        .step.active { color: #1da1f2; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏗️ Topology — 3D Urban Modeling</h1>
            <p style="color: #657786; margin-top: 5px;">Генерация 3D моделей городских территорий</p>
        </header>

        <div class="grid">
            <div class="panel">
                <h2 style="font-size: 18px; margin-bottom: 20px;">Новая задача</h2>

                <div class="form-group">
                    <label>Долгота (lon)</label>
                    <input type="number" id="lon" value="58.050485" step="0.001" min="-180" max="180">
                </div>

                <div class="form-group">
                    <label>Широта (lat)</label>
                    <input type="number" id="lat" value="56.343056" step="0.001" min="-90" max="90">
                </div>

                <div class="form-group">
                    <label>Радиус (м)</label>
                    <input type="number" id="radius" value="500" min="500" max="3000">
                </div>

                <div class="form-group">
                    <label>Детализация</label>
                    <select id="detail">
                        <option value="LOD1">LOD1 — Базовая</option>
                        <option value="LOD2">LOD2 — Средняя</option>
                    </select>
                </div>

                <button onclick="createJob()">Создать задачу →</button>

                <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #38444d; font-size: 12px; color: #657786;">
                    <div id="status" style="margin: 5px 0;">Готово</div>
                </div>
            </div>

            <div class="panel">
                <h2 style="font-size: 18px; margin-bottom: 15px;">Мои задачи</h2>
                <div class="jobs-list" id="jobsList">
                    <div style="text-align: center; color: #657786; padding: 40px;">Загрузка...</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2 style="font-size: 18px; margin-bottom: 15px;">Просмотр модели</h2>
            <div class="viewer" id="viewer">
                <div class="viewer-placeholder">Выберите задачу для просмотра</div>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #657786;" id="viewerInfo"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@r128/examples/js/loaders/GLTFLoader.js"></script>

    <script>
        const API = 'http://localhost:8000';
        let currentJobId = null;
        let scene, camera, renderer;

        async function createJob() {
            const lon = parseFloat(document.getElementById('lon').value);
            const lat = parseFloat(document.getElementById('lat').value);
            const radius = parseFloat(document.getElementById('radius').value);
            const detail = document.getElementById('detail').value;

            document.getElementById('status').textContent = 'Создание задачи...';

            try {
                const res = await fetch(`${API}/api/jobs`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        center: { lon, lat },
                        radius_m: radius,
                        detail: detail
                    })
                });

                if (!res.ok) throw new Error(await res.text());
                const job = await res.json();

                document.getElementById('status').textContent = `✅ Задача создана: ${job.id}`;
                loadJobs();
            } catch (e) {
                document.getElementById('status').textContent = `❌ Ошибка: ${e.message}`;
            }
        }

        async function loadJobs() {
            try {
                const res = await fetch(`${API}/api/jobs`);
                if (!res.ok) return;

                const jobs = await res.json();
                const html = jobs.length === 0
                    ? '<div style="text-align: center; color: #657786; padding: 40px;">Нет задач</div>'
                    : jobs.map(job => `
                        <div class="job-item" onclick="selectJob('${job.id}')">
                            <div class="job-header">
                                <span class="job-title">${job.center.lon.toFixed(3)}, ${job.center.lat.toFixed(3)}</span>
                                <span class="job-status status-${job.status}">${job.status}</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${job.status === 'pending' ? '50' : '100'}%"></div>
                            </div>
                            ${job.steps.length > 0 ? `
                                <div style="font-size: 11px; color: #657786;">
                                    ${job.steps.map(s => `<div class="step ${s.status === 'pending' ? 'active' : ''}">${s.step_name}</div>`).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `).join('');

                document.getElementById('jobsList').innerHTML = html;
            } catch (e) {
                console.error(e);
            }
        }

        async function selectJob(jobId) {
            currentJobId = jobId;

            try {
                const res = await fetch(`${API}/api/models/${jobId}/files`);
                if (!res.ok) return;

                const data = await res.json();
                const files = data.files || [];
                const glbFile = files.find(f => f.storage_key.includes('.glb'));

                if (glbFile) {
                    loadModel(glbFile.download_url);
                    document.getElementById('viewerInfo').innerHTML = `
                        <a href="${glbFile.download_url}" class="download-link">📥 Скачать GLB</a>
                    `;
                } else {
                    document.getElementById('viewerInfo').innerHTML = '<span style="color: #657786;">Модель еще не готова</span>';
                }
            } catch (e) {
                console.error(e);
            }
        }

        function loadModel(url) {
            if (!scene) initViewer();

            const loader = new THREE.GLTFLoader();
            loader.load(url, (gltf) => {
                scene.children = scene.children.filter(c => c.isLight || c.isCamera);
                scene.add(gltf.scene);

                const box = new THREE.Box3().setFromObject(gltf.scene);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);

                camera.position.copy(center);
                camera.position.z += maxDim * 1.5;
                camera.lookAt(center);

                render();
            });
        }

        function initViewer() {
            const container = document.getElementById('viewer');
            container.innerHTML = '';

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const light = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(light);

            const directional = new THREE.DirectionalLight(0xffffff, 0.8);
            directional.position.set(10, 20, 10);
            scene.add(directional);

            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });
        }

        function render() {
            renderer.render(scene, camera);
        }

        loadJobs();
        setInterval(loadJobs, 2000);
    </script>
</body>
</html>"""
    return Response(content=html, media_type="text/html")

app.mount("/viewer", StaticFiles(directory=VIEWER_DIR), name="viewer")
app.mount("/citymap", StaticFiles(directory=CITYMAP_DIR), name="citymap")
app.mount("/panorama", StaticFiles(directory=PANORAMA_DIR), name="panorama")

# Данные карты города (city.pmtiles + пирамида terrain-RGB) — общегородские
# артефакты, не привязанные к конкретной задаче (в отличие от /models/*), их
# путь настраивается `CITYMAP_DATA_DIR` (тот же приём, что `TOPOLOGY_STORAGE_
# ROOT` у `tasks.pipeline_tasks.get_storage`) — реальная городская выгрузка
# в этой среде не строится (см. docs/citymap.md: сетевая политика песочницы
# блокирует Geofabrik/Overpass), поэтому по умолчанию каталог не примонтирован
# и `/citymap-data/*` отдаёт 404, а не падает на старте.
_citymap_data_root = os.environ.get("CITYMAP_DATA_DIR")
if _citymap_data_root and Path(_citymap_data_root).is_dir():
    app.mount("/citymap-data", StaticFiles(directory=_citymap_data_root), name="citymap-data")


def _download_url(job_id: uuid.UUID, storage_key: str) -> str:
    return f"/models/{job_id}/download?key={quote(storage_key, safe='')}"


def _viewer_url(job_id: uuid.UUID, download_url: str) -> str:
    return f"/viewer/index.html?model={quote(download_url, safe='')}&job={job_id}"


def _tileset_viewer_url(job_id: uuid.UUID, download_url: str) -> str:
    """Шаг 2.11, п. 2-3: потоковый режим вьюера — `?tileset=` вместо
    `?model=`; `job` нужен вьюеру, чтобы строить URL отдельных тайлов через
    тот же `/models/{job}/download?key=...`, не полагаясь на структуру
    Storage-ключей (см. `web/viewer/index.html`)."""
    return f"/viewer/index.html?tileset={quote(download_url, safe='')}&job={job_id}"


def _job_to_out(job: store.Job) -> JobOut:
    return JobOut(
        id=job.id,
        center=Center(lon=job.center_lon, lat=job.center_lat),
        radius_m=job.radius_m,
        layers=job.layers,
        detail=job.detail,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        steps=[
            JobStepOut(
                step_name=s.step_name,
                step_order=s.step_order,
                status=s.status,
                started_at=s.started_at,
                finished_at=s.finished_at,
                error_message=s.error_message,
                result=s.result,
            )
            for s in job.steps
        ],
    )


def _user_to_out(user: auth_store.User) -> UserOut:
    return UserOut(
        id=user.id, email=user.email, role=user.role,
        closed_contour_access=user.closed_contour_access, created_at=user.created_at,
    )


def get_current_user(
    conn=Depends(get_connection), authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> auth_store.User | None:
    """Опциональная авторизация (Шаг 4.10, п. 1 / Шаг 4.11, п. 2):
    `Authorization: Bearer <токен сессии>` ИЛИ `X-API-Key: <ключ>` — `None`
    без заголовков или с невалидным значением, так `POST /jobs` остаётся
    рабочим для анонимных запросов (существующее поведение Шага 1.3, ни
    один старый тест не передаёт эти заголовки), а не требует их
    принудительно."""
    if authorization and authorization.startswith("Bearer "):
        user = auth_store.get_user_by_token(conn, authorization.removeprefix("Bearer "))
        if user is not None:
            return user
    if x_api_key:
        return auth_integrations.get_user_by_api_key(conn, x_api_key)
    return None


def require_user(user: auth_store.User | None = Depends(get_current_user)) -> auth_store.User:
    if user is None:
        raise HTTPException(status_code=401, detail="требуется авторизация")
    return user


def require_admin(user: auth_store.User = Depends(require_user)) -> auth_store.User:
    if user.role != auth_store.ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="требуется роль администратора")
    return user


def require_closed_contour_access(user: auth_store.User = Depends(require_user)) -> auth_store.User:
    if not user.closed_contour_access:
        raise HTTPException(status_code=403, detail="требуется доступ к закрытому контуру")
    return user


@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(payload: RegisterRequest, conn=Depends(get_connection)) -> TokenResponse:
    """Самостоятельная регистрация всегда даёт роль «просмотр» (Шаг 4.10,
    п. 1) — повышение роли делает администратор через `PATCH
    /auth/users/{id}/role`, не сам пользователь при регистрации."""
    try:
        user = auth_store.create_user(conn, email=payload.email, password=payload.password)
    except auth_store.EmailAlreadyRegisteredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = auth_store.create_session(conn, user.id)
    return TokenResponse(token=token, user=_user_to_out(user))


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, conn=Depends(get_connection)) -> TokenResponse:
    user = auth_store.authenticate(conn, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="неверный email или пароль")
    token = auth_store.create_session(conn, user.id)
    return TokenResponse(token=token, user=_user_to_out(user))


@app.post("/auth/logout", status_code=204)
def logout(authorization: str = Header(...), conn=Depends(get_connection)) -> Response:
    if authorization.startswith("Bearer "):
        auth_store.revoke_session(conn, authorization.removeprefix("Bearer "))
    return Response(status_code=204)


@app.get("/auth/me", response_model=UserOut)
def get_me(user: auth_store.User = Depends(require_user)) -> UserOut:
    return _user_to_out(user)


@app.patch("/auth/users/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: uuid.UUID, payload: RoleUpdateRequest,
    _admin: auth_store.User = Depends(require_admin), conn=Depends(get_connection),
) -> UserOut:
    if payload.role not in auth_store.ROLES:
        raise HTTPException(status_code=422, detail=f"неизвестная роль {payload.role!r}, ожидается одна из {auth_store.ROLES}")
    updated = auth_store.get_user_by_id(conn, user_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="пользователь не найден")
    auth_store.set_user_role(conn, user_id, role=payload.role, closed_contour_access=payload.closed_contour_access)
    return _user_to_out(auth_store.get_user_by_id(conn, user_id))


@app.post("/jobs", response_model=JobCreateResponse, status_code=201)
def create_job(
    payload: JobCreateRequest, conn=Depends(get_connection),
    user: auth_store.User | None = Depends(get_current_user),
) -> JobCreateResponse:
    job = store.create_job(
        conn,
        center_lon=payload.center.lon,
        center_lat=payload.center.lat,
        radius_m=payload.radius_m,
        layers=payload.layers,
        detail=payload.detail,
        step_names=DEFAULT_STEP_NAMES,
    )
    if user is not None:
        auth_store.record_job_ownership(conn, job.id, user.id)
    enqueue_job(str(job.id))
    return JobCreateResponse(id=job.id, status=store.get_job(conn, job.id).status)


@app.get("/projects", response_model=list[ProjectOut])
def list_projects(user: auth_store.User = Depends(require_user), conn=Depends(get_connection)) -> list[ProjectOut]:
    """«Проекты: участки, модели, версии...» (Шаг 4.10, п. 2) — список
    задач текущего пользователя. Версии/ЖК/рендеры/ТУ-запросы каждой задачи
    честно не сведены в один ответ — соответствующие данные (Шаги 3.6-3.12,
    4.5) не хранят ссылку на `jobs.id` в своих схемах, это отдельная
    интеграционная работа, не часть этого шага (см. `docs/dashboard.md`)."""
    projects = []
    for job_id in auth_store.list_job_ids_for_user(conn, user.id):
        job = store.get_job(conn, job_id)
        if job is None:
            continue
        projects.append(ProjectOut(
            id=job.id, center=Center(lon=job.center_lon, lat=job.center_lat), radius_m=job.radius_m,
            detail=job.detail, status=job.status, created_at=job.created_at, updated_at=job.updated_at,
        ))
    return projects


@app.get("/me/notifications", response_model=list[NotificationOut])
def list_my_notifications(
    unread_only: bool = False, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> list[NotificationOut]:
    return [
        NotificationOut(id=n.id, message=n.message, job_id=n.job_id, created_at=n.created_at, read_at=n.read_at)
        for n in auth_store.list_notifications(conn, user.id, unread_only=unread_only)
    ]


@app.post("/me/notifications/{notification_id}/read", status_code=204)
def mark_notification_read(
    notification_id: int, _user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> Response:
    auth_store.mark_notification_read(conn, notification_id)
    return Response(status_code=204)


def _job_bbox_wgs84(job: store.Job) -> tuple[float, float, float, float]:
    zone = pick_msk59_zone(job.center_lon)
    x, y, zone = wgs84_to_msk59(job.center_lon, job.center_lat, zone)
    corners = [
        msk59_to_wgs84(x - job.radius_m, y - job.radius_m, zone),
        msk59_to_wgs84(x + job.radius_m, y - job.radius_m, zone),
        msk59_to_wgs84(x - job.radius_m, y + job.radius_m, zone),
        msk59_to_wgs84(x + job.radius_m, y + job.radius_m, zone),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


@app.get("/projects/{job_id}/closed-contour", response_model=ClosedContourResponse)
def get_closed_contour_summary(
    job_id: uuid.UUID, _user: auth_store.User = Depends(require_closed_contour_access), conn=Depends(get_connection),
) -> ClosedContourResponse:
    """Демонстрация реального действия флага `closed_contour_access` (Шаг
    4.10, п. 1: «доступ к закрытому контуру» — не должность, а право на
    данные сетей/изысканий, см. `docs/plan.md` п. 10.4): настоящий запрос
    официальных ограничений (Шаг 3.1) по реальному bbox задачи, а не
    заглушка."""
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    constraints_store.ensure_schema(conn)
    zones = constraints_store.find_zones(conn, _job_bbox_wgs84(job))
    return ClosedContourResponse(
        job_id=job.id,
        zones=[ClosedContourZoneOut(zone_type=z.zone_type, status=z.status, registry_number=z.registry_number) for z in zones],
    )


@app.post("/jobs/{job_id}/share", response_model=ShareLinkResponse)
def share_job(
    job_id: uuid.UUID, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> ShareLinkResponse:
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    owner_id = auth_store.get_job_owner(conn, job_id)
    if owner_id is not None and owner_id != user.id:
        raise HTTPException(status_code=403, detail="только владелец задачи может создать публичную ссылку")
    token = auth_store.create_public_link(conn, job_id)
    return ShareLinkResponse(token=token, public_url=f"/public/{token}")


@app.get("/public/{token}")
def open_public_link(token: str, conn=Depends(get_connection)) -> RedirectResponse:
    """Публичная ссылка на просмотр модели (Шаг 4.10, п. 4) — честно: не
    новая граница доступа (см. докстринг `auth.store.create_public_link`),
    а стабильный, отзываемый адрес прямо во вьюер."""
    job_id = auth_store.resolve_public_link(conn, token)
    if job_id is None:
        raise HTTPException(status_code=404, detail="ссылка не найдена или отозвана")
    files = get_job_files(job_id, conn=conn)
    viewable = next((f for f in files.files if f.viewer_url), None)
    if viewable is None:
        raise HTTPException(status_code=404, detail="для задачи ещё нет готовой модели для просмотра")
    return RedirectResponse(url=viewable.viewer_url)


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, conn=Depends(get_connection)) -> JobOut:
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    return _job_to_out(job)


def _file_out(job_id: uuid.UUID, step_name: str, storage_key: str) -> FileOut:
    download_url = _download_url(job_id, storage_key)
    viewer_url = None
    if storage_key.endswith(".glb"):
        viewer_url = _viewer_url(job_id, download_url)
    elif storage_key.endswith("tileset.json"):
        viewer_url = _tileset_viewer_url(job_id, download_url)
    return FileOut(step_name=step_name, storage_key=storage_key, download_url=download_url, viewer_url=viewer_url)


@app.get("/models/{job_id}/files", response_model=FilesResponse)
def get_job_files(job_id: uuid.UUID, conn=Depends(get_connection)) -> FilesResponse:
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    files: list[FileOut] = []
    for s in job.steps:
        if s.status != store.STATUS_DONE or not s.result:
            continue
        if "storage_key" in s.result:
            files.append(_file_out(job.id, s.step_name, s.result["storage_key"]))
        # Форматы, не зависящие от IFC-схемы (Шаг 2.10, п. 2: LandXML/
        # CityJSON/DXF) - строятся в `assemble_ifc` ОДИН раз, их ключи лежат
        # прямо в `s.result`, не под `schemas[schema_name]` - тот же приём
        # раскрытия `*_storage_key`, что и ниже для схемных ключей.
        for key, value in s.result.items():
            if key != "storage_key" and key.endswith("_storage_key"):
                prefix = key.removesuffix("_storage_key")
                files.append(_file_out(job.id, f"{s.step_name}:{prefix}", value))
        for schema_name, schema_result in s.result.get("schemas", {}).items():
            if "storage_key" in schema_result:
                files.append(_file_out(job.id, f"{s.step_name}:{schema_name}", schema_result["storage_key"]))
            # Каркасная/внутриквартальная сеть отдельными файлами (Шаг 2.4,
            # п. 4) - тот же schema_result, но под своими ключами
            # (`roads_backbone_storage_key`/`roads_internal_storage_key`,
            # `assemble_ifc`), не единственным `storage_key`.
            for key, value in schema_result.items():
                if key != "storage_key" and key.endswith("_storage_key"):
                    prefix = key.removesuffix("_storage_key")
                    files.append(_file_out(job.id, f"{s.step_name}:{schema_name}:{prefix}", value))
    return FilesResponse(job_id=job.id, status=job.status, files=files)


@app.get("/models/{job_id}/download")
def download_job_file(job_id: uuid.UUID, key: str, conn=Depends(get_connection)) -> Response:
    """Отдать байты результата шага из хранилища (Шаг 1.9, п. 3: ссылка на
    модель из журнала задачи ведёт сюда) — префикс `key` привязан к job_id,
    чтобы одна задача не могла скачать файлы другой."""
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    if not key.startswith(f"jobs/{job_id}/"):
        raise HTTPException(status_code=404, detail="файл не найден")
    try:
        data = get_storage().download(key)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="файл не найден") from None
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    if key.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif key.endswith(".ifc"):
        media_type = "application/x-step"
    return Response(content=data, media_type=media_type)


@app.post("/me/api-keys", response_model=ApiKeyCreateResponse, status_code=201)
def create_my_api_key(
    payload: ApiKeyCreateRequest, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> ApiKeyCreateResponse:
    """«Ключи доступа» (Шаг 4.11, п. 2) — для машинных клиентов, отдельно
    от сессионных токенов (Шаг 4.10): передаются заголовком `X-API-Key`
    (см. `get_current_user`), не истекают по времени, отзываются вручную."""
    raw_key, meta = auth_integrations.create_api_key(conn, user.id, label=payload.label)
    return ApiKeyCreateResponse(id=meta.id, label=meta.label, created_at=meta.created_at, key=raw_key)


@app.get("/me/api-keys", response_model=list[ApiKeyOut])
def list_my_api_keys(user: auth_store.User = Depends(require_user), conn=Depends(get_connection)) -> list[ApiKeyOut]:
    return [
        ApiKeyOut(id=k.id, label=k.label, created_at=k.created_at, revoked_at=k.revoked_at)
        for k in auth_integrations.list_api_keys(conn, user.id)
    ]


@app.delete("/me/api-keys/{key_id}", status_code=204)
def revoke_my_api_key(
    key_id: int, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> Response:
    if not auth_integrations.revoke_api_key(conn, key_id, user.id):
        raise HTTPException(status_code=404, detail="ключ не найден")
    return Response(status_code=204)


@app.post("/me/webhooks", response_model=WebhookOut, status_code=201)
def register_my_webhook(
    payload: WebhookRegisterRequest, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> WebhookOut:
    """«Вебхуки о готовности задачи» (Шаг 4.11, п. 4) — доставка настоящим
    HTTP POST при переходе задачи в терминальный статус, см.
    `tasks.pipeline_tasks._notify_job_completion`."""
    try:
        webhook = auth_integrations.register_webhook(conn, user.id, url=payload.url)
    except auth_integrations.InvalidWebhookUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WebhookOut(id=webhook.id, url=webhook.url, created_at=webhook.created_at)


@app.get("/me/webhooks", response_model=list[WebhookOut])
def list_my_webhooks(user: auth_store.User = Depends(require_user), conn=Depends(get_connection)) -> list[WebhookOut]:
    return [WebhookOut(id=w.id, url=w.url, created_at=w.created_at) for w in auth_integrations.list_webhooks(conn, user.id)]


@app.delete("/me/webhooks/{webhook_id}", status_code=204)
def revoke_my_webhook(
    webhook_id: int, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> Response:
    if not auth_integrations.revoke_webhook(conn, webhook_id, user.id):
        raise HTTPException(status_code=404, detail="вебхук не найден")
    return Response(status_code=204)


@app.post("/jobs/{job_id}/export-to-pilot-bim", response_model=ExchangeFolderExportResponse)
def export_job_to_pilot_bim(
    job_id: uuid.UUID, user: auth_store.User = Depends(require_user), conn=Depends(get_connection),
) -> ExchangeFolderExportResponse:
    """Выгрузка в Pilot-BIM через папку обмена (Шаг 4.11, п. 3) — см.
    докстринг `export.exchange_folder`: настоящая интеграция с API
    Pilot-BIM не выполнена (проприетарная платформа, нет доступа), план
    явно предлагает эту альтернативу. Путь папки — `PILOT_BIM_EXCHANGE_
    DIR` (тот же приём конфигурации, что `TOPOLOGY_STORAGE_ROOT`/
    `CITYMAP_DATA_DIR`); без переменной — понятная ошибка, а не падение."""
    exchange_dir = os.environ.get("PILOT_BIM_EXCHANGE_DIR")
    if not exchange_dir:
        raise HTTPException(status_code=503, detail="папка обмена с Pilot-BIM не настроена (PILOT_BIM_EXCHANGE_DIR)")

    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    owner_id = auth_store.get_job_owner(conn, job_id)
    if owner_id is not None and owner_id != user.id:
        raise HTTPException(status_code=403, detail="только владелец задачи может выгрузить её в Pilot-BIM")

    files = get_job_files(job_id, conn=conn)
    ifc_keys = [f.storage_key for f in files.files if f.storage_key.endswith(".ifc")]
    if not ifc_keys:
        raise HTTPException(status_code=404, detail="для задачи ещё нет готовой IFC-модели")

    result = export_job_to_exchange_folder(get_storage(), job_id, ifc_keys, Path(exchange_dir))
    return ExchangeFolderExportResponse(
        job_folder=str(result.job_folder),
        exported_files=[str(p) for p in result.exported_files],
        missing_keys=result.missing_keys,
    )


# ===== ОТКРЫТЫЕ ЭНДПОИНТЫ ДЛЯ ВЕБ-ИНТЕРФЕЙСА (БЕЗ АВТОРИЗАЦИИ) =====

@app.get("/api/jobs", response_model=list[JobOut])
def get_all_jobs(conn=Depends(get_connection)) -> list[JobOut]:
    """Получить все задачи (без авторизации для веб-интерфейса)"""
    all_jobs = store.list_all_jobs(conn)
    return [
        JobOut(
            id=job.id,
            center=Center(lon=job.center_lon, lat=job.center_lat),
            radius_m=job.radius_m,
            layers=job.layers,
            detail=job.detail,
            status=job.status,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            steps=[
                JobStepOut(
                    step_name=step.step_name,
                    step_order=step.step_order,
                    status=step.status,
                    started_at=step.started_at,
                    finished_at=step.finished_at,
                    error_message=step.error_message,
                    result=step.result,
                )
                for step in job.steps
            ],
        )
        for job in all_jobs
    ]


@app.post("/api/jobs", response_model=JobCreateResponse)
def create_job_open(req: JobCreateRequest, conn=Depends(get_connection)):
    """Создать задачу (без авторизации для веб-интерфейса)"""
    job = store.Job.new_job(
        center_lon=req.center.lon,
        center_lat=req.center.lat,
        radius_m=req.radius_m,
        layers=req.layers,
        detail=req.detail,
    )
    
    # Создать default пользователя если не существует
    default_user = auth_store.ensure_default_user(conn)
    
    job_id = store.insert_job(conn, job, owner_id=default_user.id)
    enqueue_job(job_id)
    
    return JobCreateResponse(id=job_id, status="pending")


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job_open(job_id: uuid.UUID, conn=Depends(get_connection)) -> JobOut:
    """Получить задачу (без авторизации)"""
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    return JobOut(
        id=job.id,
        center=Center(lon=job.center_lon, lat=job.center_lat),
        radius_m=job.radius_m,
        layers=job.layers,
        detail=job.detail,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        steps=[
            JobStepOut(
                step_name=step.step_name,
                step_order=step.step_order,
                status=step.status,
                started_at=step.started_at,
                finished_at=step.finished_at,
                error_message=step.error_message,
                result=step.result,
            )
            for step in job.steps
        ],
    )


@app.get("/api/models/{job_id}/files", response_model=FilesResponse)
def get_files_open(job_id: uuid.UUID, conn=Depends(get_connection)) -> FilesResponse:
    """Получить файлы задачи (без авторизации)"""
    job = store.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена")
    
    files = get_job_files(job_id, conn=conn)
    return files
