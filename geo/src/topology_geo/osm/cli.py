"""CLI-обёртка над журналом загрузок OSM (Шаг 1.1, п. 4).

Используется скриптами импорта (`geo/scripts/import_osm.sh`,
`geo/scripts/update_osm_replication.sh`), чтобы не дублировать логику
подключения/записи на bash.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from topology_geo.devcheck import load_environment_config
from topology_geo.osm.import_log import ImportLogEntry, ensure_schema, latest_import, record_import


def _connect(config):
    import psycopg

    return psycopg.connect(config.postgres.dsn)


def _cmd_record_import(args: argparse.Namespace) -> int:
    config = load_environment_config()
    data_timestamp = datetime.fromisoformat(args.data_timestamp)
    if data_timestamp.tzinfo is None:
        data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)

    with _connect(config) as conn:
        ensure_schema(conn)
        row_id = record_import(
            conn,
            ImportLogEntry(
                source_name=args.source_name,
                source_file=args.source_file,
                data_timestamp=data_timestamp,
                notes=args.notes,
            ),
        )
    print(f"osm_import_log id={row_id}")
    return 0


def _cmd_latest_import(args: argparse.Namespace) -> int:
    config = load_environment_config()
    with _connect(config) as conn:
        ensure_schema(conn)
        entry = latest_import(conn, args.source_name)
    if entry is None:
        print(f"Нет записей для источника {args.source_name!r}")
        return 1
    print(f"{entry.source_name}: данные на {entry.data_timestamp.isoformat()} (файл: {entry.source_file})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record-import", help="Записать факт импорта в osm_import_log")
    record.add_argument("--source-name", required=True)
    record.add_argument("--source-file", required=True)
    record.add_argument("--data-timestamp", required=True, help="ISO 8601, например 2026-09-20")
    record.add_argument("--notes", default=None)
    record.set_defaults(func=_cmd_record_import)

    latest = sub.add_parser("latest-import", help="Показать последнюю запись по источнику")
    latest.add_argument("--source-name", required=True)
    latest.set_defaults(func=_cmd_latest_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
