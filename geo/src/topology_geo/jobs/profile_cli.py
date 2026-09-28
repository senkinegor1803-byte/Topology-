"""Профилирование конвейера по времени и памяти (Шаг 2.13, п. 1: «замер
времени по шагам на 3 участках, профилирование узких мест»).

Время по шагу уже пишется для КАЖДОГО реального прогона (`store.start_step`/
`finish_step` — `started_at`/`finished_at` в `job_steps`, Шаг 1.3) — здесь не
дублируется, а читается обратно из БД. Единственное, чего не было —
память: отдельный поток раз в `--sample-interval-s` читает `VmRSS` текущего
процесса из `/proc/self/status` (без новой зависимости) и пишет замеры в
CSV НЕМЕДЛЕННО (`flush()` после каждой строки) — если процесс упадёт от
OOM-killer (реальный, документированный риск в этой среде, см.
`docs/dev-tree.md`, Шаги 2.4/2.6/2.10), сам процесс Python не успеет ничего
записать в момент SIGKILL, но всё, что было записано ДО него, останется на
диске; шаги `job_steps` тоже уже закоммичены (`autocommit=True`,
`devcheck.load_environment_config`) — значит, что именно выполнялось на
момент смерти, видно и без итогового отчёта.

Поэтому две подкоманды:
  run    — создать и выполнить задачу с фоновым замером памяти.
  report — по job_id (+ файлу замеров памяти, если он есть) собрать отчёт
           даже после того, как `run` был убит OOM-killer'ом.

Использование:
    python -m topology_geo.jobs.profile_cli run --lon 56.342231 --lat 58.049986 \\
        --radius-m 500 --memory-log mem.csv --report report.json
    # если `run` не завершился (OOM) - report всё равно можно получить:
    python -m topology_geo.jobs.profile_cli report --job-id <id> \\
        --memory-log mem.csv --report report.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import psycopg

from topology_geo.devcheck import load_environment_config
from topology_geo.jobs import store
from topology_geo.jobs.pipeline import run_all_pending_steps
from topology_geo.jobs.steps import DEFAULT_PIPELINE
from topology_geo.storage import FileSystemObjectStorage, ObjectStorage


def _connect() -> psycopg.Connection:
    config = load_environment_config()
    return psycopg.connect(config.postgres.dsn, autocommit=True)


def _read_rss_kb() -> int | None:
    """`VmRSS` текущего процесса из `/proc/self/status` (Linux, без новой
    зависимости — тот же приём, что использовался при ручном диагностике
    OOM этой сессии через `ps`/`dmesg`, только теперь как код, а не разовая
    команда)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


class _MemorySampler:
    """Фоновый поток, пишущий `timestamp_unix,rss_kb` в CSV раз в
    `interval_s` — построчно с `flush()`, чтобы данные пережили SIGKILL
    основного процесса."""

    def __init__(self, path: Path, interval_s: float) -> None:
        self._path = path
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with open(self._path, "w") as f:
            f.write("timestamp_unix,rss_kb\n")
            f.flush()
            while not self._stop.is_set():
                rss = _read_rss_kb()
                if rss is not None:
                    f.write(f"{time.time()},{rss}\n")
                    f.flush()
                self._stop.wait(self._interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_s * 3)


def _load_memory_samples(path: Path | str) -> list[tuple[float, int]]:
    path = Path(path)
    if not path.is_file():
        return []
    samples: list[tuple[float, int]] = []
    with open(path) as f:
        next(f, None)  # заголовок
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            samples.append((float(parts[0]), int(parts[1])))
    return samples


def _peak_rss_in_window(samples: list[tuple[float, int]], start_ts: float, end_ts: float) -> int | None:
    in_window = [rss for ts, rss in samples if start_ts <= ts <= end_ts]
    return max(in_window) if in_window else None


def build_report(job: store.Job, memory_samples: list[tuple[float, int]]) -> dict:
    steps_report = []
    for s in job.steps:
        duration_s = None
        peak_rss_kb = None
        if s.started_at is not None:
            end = s.finished_at or datetime.now(s.started_at.tzinfo)
            duration_s = (end - s.started_at).total_seconds()
            peak_rss_kb = _peak_rss_in_window(memory_samples, s.started_at.timestamp(), end.timestamp())
        steps_report.append({
            "step_name": s.step_name,
            "status": s.status,
            "duration_s": duration_s,
            "peak_rss_kb": peak_rss_kb,
            "error_message": s.error_message,
        })

    overall_peak_rss_kb = max((r for _, r in memory_samples), default=None)
    overall_duration_s = sum(r["duration_s"] for r in steps_report if r["duration_s"] is not None) or None

    return {
        "job_id": str(job.id),
        "center": {"lon": job.center_lon, "lat": job.center_lat},
        "radius_m": job.radius_m,
        "status": job.status,
        "steps": steps_report,
        "overall_duration_s": overall_duration_s,
        "overall_peak_rss_kb": overall_peak_rss_kb,
    }


def _print_report(report: dict) -> None:
    print(f"Задача {report['job_id']} r={report['radius_m']} м — статус: {report['status']}")
    print(f"{'шаг':<24}{'статус':<10}{'время, с':>10}{'RSS, МБ':>12}")
    for s in report["steps"]:
        duration = f"{s['duration_s']:.1f}" if s["duration_s"] is not None else "—"
        rss_mb = f"{s['peak_rss_kb'] / 1024:.0f}" if s["peak_rss_kb"] is not None else "—"
        print(f"{s['step_name']:<24}{s['status']:<10}{duration:>10}{rss_mb:>12}")
    overall_rss = report["overall_peak_rss_kb"]
    print(f"Итого: {report['overall_duration_s'] or 0:.1f} с, пик RSS "
          f"{(overall_rss / 1024) if overall_rss else 0:.0f} МБ")


def _cmd_run(args: argparse.Namespace) -> int:
    conn = _connect()
    store.ensure_schema(conn)
    storage: ObjectStorage = FileSystemObjectStorage(args.storage_root) if args.storage_root else FileSystemObjectStorage(
        str(Path(args.report).with_suffix("")) + "_storage"
    )

    job = store.create_job(
        conn, center_lon=args.lon, center_lat=args.lat, radius_m=args.radius_m,
        layers=[], detail="LOD1", step_names=list(DEFAULT_PIPELINE),
    )
    print(f"Создана задача {job.id}")

    sampler = _MemorySampler(Path(args.memory_log), args.sample_interval_s)
    sampler.start()
    try:
        run_all_pending_steps(conn, storage, job.id, DEFAULT_PIPELINE)
    finally:
        sampler.stop()

    job = store.get_job(conn, job.id)
    memory_samples = _load_memory_samples(Path(args.memory_log))
    report = build_report(job, memory_samples)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    conn = _connect()
    job = store.get_job(conn, uuid.UUID(args.job_id))
    if job is None:
        print(f"задача {args.job_id} не найдена", flush=True)
        return 1
    memory_samples = _load_memory_samples(Path(args.memory_log)) if args.memory_log else []
    report = build_report(job, memory_samples)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="создать и выполнить задачу с замером памяти")
    run_p.add_argument("--lon", type=float, required=True)
    run_p.add_argument("--lat", type=float, required=True)
    run_p.add_argument("--radius-m", type=float, required=True)
    run_p.add_argument("--storage-root", default=None, help="каталог FileSystemObjectStorage (по умолчанию рядом с --report)")
    run_p.add_argument("--memory-log", required=True, help="CSV для замеров памяти (переживает SIGKILL)")
    run_p.add_argument("--sample-interval-s", type=float, default=0.5)
    run_p.add_argument("--report", required=True, help="путь для итогового JSON-отчёта")
    run_p.set_defaults(func=_cmd_run)

    report_p = sub.add_parser("report", help="собрать отчёт по уже созданной/упавшей задаче")
    report_p.add_argument("--job-id", required=True)
    report_p.add_argument("--memory-log", default=None)
    report_p.add_argument("--report", required=True)
    report_p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
