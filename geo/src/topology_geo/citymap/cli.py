"""CLI: собрать пирамиду тайлов рельефа terrain-RGB (Шаг 2.12, п. 2) из
локального GeoTIFF в каталог `{z}/{x}/{y}.png` (стандартная раскладка XYZ,
её напрямую понимает источник `raster` MapLibre GL — `tiles: ["…/{z}/{x}/
{y}.png"]`, отдельный контейнер вроде PMTiles не нужен).

Источник — ЛЮБОЙ GeoTIFF (произвольная исходная CRS, `render_terrain_rgb_
tile` сам репроецирует) — например уже подготовленный `relief.tif` задачи,
реальный DEM с TessaDEM/SRTM и т.п. Само скачивание TessaDEM для полного
города в этом окружении недоступно (см. `docs/citymap.md` — сетевая
политика песочницы блокирует Geofabrik/Overpass; `api.opentopodata.org`
доступен и использовался для точечных прогонов, но не для городского
покрытия целиком) — CLI параметризован по входному файлу и не завязан на
конкретный источник."""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio

from topology_geo.citymap.terrain_rgb import render_terrain_rgb_tile, tiles_covering_bbox
from topology_geo.relief.service import Grid


def build_terrain_tile_pyramid(
    dem_path: str | Path,
    output_dir: str | Path,
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    min_zoom: int,
    max_zoom: int,
    tile_size: int = 256,
) -> int:
    """Собрать все тайлы `[min_zoom, max_zoom]`, пересекающие bbox WGS-84, в
    `output_dir/{z}/{x}/{y}.png`. Возвращает число построенных тайлов."""
    output_dir = Path(output_dir)
    with rasterio.open(dem_path) as src:
        values = src.read(1)
        grid = Grid(transform=src.transform, width=src.width, height=src.height, crs=src.crs.to_proj4())

    count = 0
    for z in range(min_zoom, max_zoom + 1):
        for x, y in tiles_covering_bbox(min_lon, min_lat, max_lon, max_lat, z):
            tile_png = render_terrain_rgb_tile(values, grid, z, x, y, tile_size=tile_size)
            tile_path = output_dir / str(z) / str(x) / f"{y}.png"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_path.write_bytes(tile_png)
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem-path", required=True, help="путь к исходному GeoTIFF рельефа")
    parser.add_argument("--output-dir", required=True, help="каталог для {z}/{x}/{y}.png")
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--min-zoom", type=int, default=8)
    parser.add_argument("--max-zoom", type=int, default=14)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    count = build_terrain_tile_pyramid(
        args.dem_path, args.output_dir,
        min_lon=args.min_lon, min_lat=args.min_lat, max_lon=args.max_lon, max_lat=args.max_lat,
        min_zoom=args.min_zoom, max_zoom=args.max_zoom, tile_size=args.tile_size,
    )
    print(f"построено тайлов: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
