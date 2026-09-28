"""CLI загрузки официальных/расчётных зон ограничений (Шаг 3.1, п. 1-2):
`load-file` — полуавтоматический режим («файл, полученным по запросу» —
GeoJSON выгрузка с НСПД/ИСОГД или подготовленный вручную)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from topology_geo.constraints.store import (
    STATUS_OFFICIAL,
    ensure_schema,
    load_zones_from_geojson,
)
from topology_geo.devcheck import load_environment_config


def _connect(config):
    import psycopg

    return psycopg.connect(config.postgres.dsn)


def _cmd_load_file(args: argparse.Namespace) -> int:
    config = load_environment_config()
    data_timestamp = datetime.fromisoformat(args.data_timestamp)
    if data_timestamp.tzinfo is None:
        data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)

    with open(args.geojson_path, encoding="utf-8") as f:
        feature_collection = json.load(f)

    with _connect(config) as conn:
        ensure_schema(conn)
        ids = load_zones_from_geojson(
            conn, feature_collection, source_name=args.source_name,
            data_timestamp=data_timestamp, default_status=args.status,
        )
    print(f"загружено зон: {len(ids)} (id: {ids})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    load_file = sub.add_parser("load-file", help="Загрузить зоны из GeoJSON-файла")
    load_file.add_argument("geojson_path", help="путь к FeatureCollection (GeoJSON)")
    load_file.add_argument("--source-name", required=True, help="например «НСПД, выгрузка по запросу»")
    load_file.add_argument("--data-timestamp", required=True, help="ISO 8601, дата актуальности данных источника")
    load_file.add_argument("--status", default=STATUS_OFFICIAL, choices=["официально", "расчётно"])
    load_file.set_defaults(func=_cmd_load_file)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
