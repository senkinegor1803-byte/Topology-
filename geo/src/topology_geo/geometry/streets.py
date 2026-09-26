"""Дороги по полосам через osm2streets (Шаг 2.3, п. 1 и 3), бордюр (п. 2)
и покрытие (п. 4).

osm2streets (A/B Street) — единственный существующий инструмент, который
по тегам OSM (`highway`, `lanes`, `sidewalk`, `parking:*`, ...) и реальной
связности узлов строит геометрию отдельных полос (проезжая часть, тротуар,
парковка, разделитель...) и площадок перекрёстков; писать это заново не
требуется и рискованно (масса краевых случаев разметки полос). Библиотека
существует только как Rust/WASM (`osm2streets-js`, `geo/osm2streets/`), в
отличие от остального конвейера (Python) — поэтому здесь она вызывается как
отдельный процесс Node.js (`geo/osm2streets/convert.mjs`), тем же приёмом,
что и системный `osm2pgsql` (Шаг 1.1): реальный внешний инструмент через
subprocess, а не переписанная на Python копия его логики.

Входные данные — не `SiteFeature` (Шаг 1.4 уже потерял связность узлов при
нормализации), а `RawRoadWay` с исходными тегами и ID узлов
(`topology_geo.osm.raw_roads`), из которых строится валидный OSM XML.

Бордюр (Шаг 2.3, п. 2) osm2streets отдельным типом полосы не отдаёт
(проверено эмпирически) — синтезируется здесь (`_build_curb_strips`) по
общей границе между объединёнными Driving- и Sidewalk-полосами одной дороги,
без бордюра на грунтовых дорогах (Шаг 2.3, п. 4, `UNPAVED_SURFACES`).

Разделитель и газон (тоже Шаг 2.3, п. 2) — в отличие от бордюра, ЭТИ два
osm2streets отдаёт сам, без синтеза: `SharedLeftTurn` (полоса разворота из
`lanes:both_ways`+`turn:lanes:both_ways=left`) и `Buffer(Verge)` (газон из
`cycleway:<сторона>:separation:<сторона>=grass_verge`) — см.
`LANE_TYPE_GRASS_VERGE` и `_surface_for_lane`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from topology_geo.coords import msk59_to_wgs84, transform_geometry_to_msk59, wgs84_to_msk59
from topology_geo.osm.raw_roads import RawRoadWay, build_osm_xml
from topology_geo.selection.clip import clip_and_localize

OSM2STREETS_DIR = Path(__file__).resolve().parents[3] / "osm2streets"
CONVERT_SCRIPT = OSM2STREETS_DIR / "convert.mjs"

# Россия (Пермский край) — правостороннее движение; влияет на то, с какой
# стороны osm2streets кладёт тротуар/парковку у дороги без явного sidewalk=*.
DEFAULT_IMPORT_OPTIONS: dict[str, object] = {
    "debug_each_step": False,
    "dual_carriageway_experiment": False,
    "sidepath_zipping_experiment": False,
    "inferred_sidewalks": True,
    "inferred_kerbs": True,
    "date_time": None,
    "override_driving_side": "Right",
    "osm2lanes": False,
}

# Вершины окружности буфера в WGS-84 (Шаг 1.4 использует круг того же
# радиуса, без запаса — запас нужен только для выборки, см. query.py).
CIRCLE_SEGMENTS = 64

# Шаг 2.3, п. 2: бордюр между проезжей частью и тротуаром - osm2streets его
# не выделяет отдельным типом полосы (проверено эмпирически: набор тегов с
# parking/cycleway/sidewalk/shoulder даёт только Driving/Sidewalk/Biking),
# синтезируется здесь (см. `_build_curb_strips`). Ширина по плану (Шаг 2.3, п. 2).
LANE_TYPE_CURB = "Curb"
CURB_WIDTH_M = 0.15

# Разделитель и газон (остаток Шага 2.3, п. 2) — в отличие от бордюра, ЭТИ
# два элемента osm2streets отдаёт САМ, отдельными типами полосы, без синтеза
# на Python (тот же принцип "источник истины типов полос остаётся у
# инструмента", что и у Driving/Sidewalk — см. docstring модуля):
# - `SharedLeftTurn` — центральная полоса для разворота/поворота налево,
#   реальный физический разделитель встречных потоков там, где нет сплошной
#   линии; получается из связки тегов `lanes:both_ways=1` +
#   `turn:lanes:both_ways=left` (проверено эмпирически, geo/osm2streets).
# - `Buffer(Verge)` — газон между проезжей частью/велодорожкой и остальным
#   профилем; получается из `cycleway:<сторона>:separation:<сторона>=
#   grass_verge` (проверено эмпирически) — то же место в OSM-схеме, что и
#   `separation=kerb/planter/jersey_barrier/...` для других физических
#   буферов, `Buffer(...)` в остальных случаях сюда же, без синтеза.
# Оба типа уже проходят через общий код полос (`LaneRibbon`, посадка на
# рельеф Шага 2.3 п. 5, экспорт в IFC) без отдельной ветки — единственное
# отличие газона от обычной полосы: покрытие (см. `_surface_for_lane`).
LANE_TYPE_GRASS_VERGE = "Buffer(Verge)"
GRASS_VERGE_SURFACE = "grass"

# `surface=*` без покрытия (вики Key:surface, раздел Unpaved, проверено
# запросом при разработке) - у таких дорог бордюра не бывает (Шаг 2.3, п. 4:
# «грунтовые дороги — без бордюров, с колеёй»).
UNPAVED_SURFACES = frozenset({
    "unpaved", "compacted", "fine_gravel", "gravel", "shells", "rock", "pebblestone",
    "ground", "dirt", "earth", "mud", "laterite", "grass", "sand", "woodchips", "snow", "ice", "salt",
})


@dataclass(frozen=True)
class LaneRibbon:
    """Полигон одной полосы (Шаг 2.3, п. 1) в локальных координатах участка."""

    osm_way_ids: tuple[int, ...]
    lane_type: str  # значение osm2streets: Driving/Sidewalk/Parking/Shoulder/Biking/
    # SharedLeftTurn/Buffer(Verge)/Buffer(...)/... (или "Curb" — наш синтез, см. ниже)
    width_m: float
    direction: str  # Fwd/Back (или "" для перекрёстков/симметричных элементов)
    polygon: Polygon
    surface: str | None = None  # тег surface=* исходной дороги как есть (Шаг 2.3, п. 4); None у бордюра - не тег


@dataclass(frozen=True)
class IntersectionArea:
    """Полигон перекрёстка/угла тротуара (Шаг 2.3, п. 1) в локальных координатах."""

    kind: str
    polygon: Polygon


@dataclass(frozen=True)
class LaneMarking:
    """Полигон элемента разметки (Шаг 2.3, п. 3: `toLaneMarkingsGeojson` —
    центральные линии, стрелки поворота из `turn:lanes`) в локальных
    координатах. Без ширины/направления — osm2streets отдаёт только тип и
    саму форму (полигон уже описывает штрих/стрелку целиком)."""

    kind: str  # "center line" | "lane arrow" (значения osm2streets)
    polygon: Polygon


@dataclass(frozen=True)
class StreetNetwork:
    """Результат `build_lane_network` (Шаг 2.3, п. 1 и 3)."""

    lanes: list[LaneRibbon]
    intersections: list[IntersectionArea]
    markings: list[LaneMarking]


def is_osm2streets_available() -> bool:
    """Есть ли в окружении `node` и установленный `osm2streets-js`
    (`npm install` в `geo/osm2streets/`). Используется и тестами (пропуск при
    отсутствии), и `jobs.steps.assemble_ifc` — как честный водопад: если
    инструмента нет, участок остаётся с полосами-заглушками Шага 1.7
    (`RoadRibbon`), не падает (тот же приём, что `NullOvertureSource`,
    Шаг 2.2, п. 2)."""
    import shutil

    return shutil.which("node") is not None and (OSM2STREETS_DIR / "node_modules" / "osm2streets-js").is_dir()


def _circle_geojson_wgs84(center_lon: float, center_lat: float, radius_m: float, zone: int) -> str:
    """Окружность буфера в WGS-84 как GeoJSON Feature — clip-полигон для
    osm2streets. Строится через буфер в МСК-59 (метры) и обратную
    репроекцию каждой вершины (тот же приём точности, что `clip.py`/
    `jobs.steps._wgs84_bbox_for_radius`), а не наивным градусным отступом."""
    center_x, center_y, _ = wgs84_to_msk59(center_lon, center_lat, zone=zone)
    circle = Point(center_x, center_y).buffer(radius_m, quad_segs=CIRCLE_SEGMENTS // 4)
    ring_lonlat = [msk59_to_wgs84(x, y, zone=zone) for x, y in circle.exterior.coords]
    return json.dumps(
        {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring_lonlat]]},
        }
    )


def _as_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if not g.is_empty and g.geom_type == "Polygon"]
    return [geom] if geom.geom_type == "Polygon" else []


def _localize_features(features: list[dict], zone: int, center_x: float, center_y: float, radius_m: float):
    """GeoJSON-фичи (WGS-84) -> список (полигон в локальных координатах,
    исходные properties) - общий шаг репроекции+обрезки для полос/
    перекрёстков/разметки (та же цепочка, что остальные слои, Шаг 1.4).

    `make_valid` перед пересечением с кругом обязателен: у osm2streets
    среди тысяч мелких полигонов разметки (штрихи центральной линии,
    стрелки) на реальных данных попадаются самопересекающиеся "бабочки"
    околонулевой площади (та же природа проблемы, что `repair_footprint`
    у контуров зданий, Шаг 1.6) - без починки `shapely` падает с
    `TopologyException` на пересечении с кругом буфера."""
    result: list[tuple[Polygon, dict]] = []
    for feature in features:
        geom_wgs84 = make_valid(shape(feature["geometry"]))
        if geom_wgs84.is_empty:
            continue
        geom_msk59 = transform_geometry_to_msk59(geom_wgs84, zone=zone)
        local_geom = clip_and_localize(geom_msk59, center_x, center_y, radius_m)
        if local_geom is None:
            continue
        props = feature.get("properties", {})
        for polygon in _as_polygons(local_geom):
            result.append((polygon, props))
    return result


def _as_lines(geom):
    if geom.is_empty:
        return []
    if geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        return [g for g in geom.geoms if not g.is_empty and g.geom_type == "LineString"]
    return [geom] if geom.geom_type == "LineString" else []


def _is_unpaved(tags: dict[str, str]) -> bool:
    return str(tags.get("surface", "")).strip().lower() in UNPAVED_SURFACES


def _surface_of(way_ids: tuple[int, ...], tags_by_way_id: dict[int, dict[str, str]]) -> str | None:
    """Тег `surface=*` дороги как есть (Шаг 2.3, п. 4) - первое найденное
    значение среди `way_ids` (в подавляющем большинстве случаев там ровно
    один way; смешанное покрытие одной полосы, слитой osm2streets из
    нескольких way, здесь не разбирается отдельно)."""
    for way_id in way_ids:
        surface = tags_by_way_id.get(way_id, {}).get("surface")
        if surface:
            return str(surface).strip() or None
    return None


def _surface_for_lane(
    lane_type: str, way_ids: tuple[int, ...], tags_by_way_id: dict[int, dict[str, str]]
) -> str | None:
    """Покрытие полосы (Шаг 2.3, п. 4) с одним честным исключением —
    газоном (`LANE_TYPE_GRASS_VERGE`, остаток Шага 2.3, п. 2): физически это
    трава, а не покрытие проезжей части, поэтому `surface=*` исходной дороги
    сюда не переносится (иначе газон получил бы `Покрытие=asphalt`, что
    неверно). Для всех остальных типов полос, включая `SharedLeftTurn`
    (разделитель — та же проезжая часть, тот же асфальт) — обычная передача
    тега дороги (`_surface_of`)."""
    if lane_type == LANE_TYPE_GRASS_VERGE:
        return GRASS_VERGE_SURFACE
    return _surface_of(way_ids, tags_by_way_id)


def _build_curb_strips(
    lanes: list[LaneRibbon], tags_by_way_id: dict[int, dict[str, str]]
) -> list[LaneRibbon]:
    """Синтезировать бордюр (Шаг 2.3, п. 2) на границе проезжей части и
    тротуара — osm2streets отдаёт только сами полосы, не бордюр между ними
    (см. docstring модуля). Граница ищется геометрически (общая линия между
    объединением всех Driving-полос и объединением всех Sidewalk-полос одной
    дороги), а не по константному смещению — так бордюр остаётся точным при
    любой форме и числе полос, которые уже посчитал osm2streets.

    Без бордюра, если: нет одновременно проезжей части и тротуара (нечего
    разделять), или дорога грунтовая (`surface` без покрытия, Шаг 2.3, п. 4
    — «грунтовые дороги без бордюров»)."""
    groups: dict[tuple[int, ...], list[LaneRibbon]] = {}
    for lane in lanes:
        groups.setdefault(lane.osm_way_ids, []).append(lane)

    curbs: list[LaneRibbon] = []
    for way_ids, group in groups.items():
        if any(_is_unpaved(tags_by_way_id.get(way_id, {})) for way_id in way_ids):
            continue

        driving = unary_union([lane.polygon for lane in group if lane.lane_type == "Driving"])
        sidewalk = unary_union([lane.polygon for lane in group if lane.lane_type == "Sidewalk"])
        if driving.is_empty or sidewalk.is_empty:
            continue

        shared_boundary = driving.boundary.intersection(sidewalk.boundary)
        lines = _as_lines(shared_boundary)
        if not lines:
            continue

        curb_geom = unary_union([line.buffer(CURB_WIDTH_M / 2, cap_style="flat") for line in lines])
        for polygon in _as_polygons(curb_geom):
            curbs.append(
                LaneRibbon(
                    osm_way_ids=way_ids, lane_type=LANE_TYPE_CURB, width_m=CURB_WIDTH_M,
                    direction="", polygon=polygon,
                )
            )

    return curbs


def _run_osm2streets(osm_xml: str, clip_geojson: str, import_options: dict[str, object]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        osm_path = tmp_path / "roads.osm.xml"
        clip_path = tmp_path / "clip.geojson"
        options_path = tmp_path / "options.json"
        osm_path.write_text(osm_xml, encoding="utf-8")
        clip_path.write_text(clip_geojson, encoding="utf-8")
        options_path.write_text(json.dumps(import_options), encoding="utf-8")

        result = subprocess.run(
            ["node", str(CONVERT_SCRIPT), str(osm_path), str(clip_path), str(options_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)


def build_lane_network(
    raw_roads: list[RawRoadWay],
    center_lon: float,
    center_lat: float,
    radius_m: float,
    zone: int,
    *,
    import_options: dict[str, object] | None = None,
) -> StreetNetwork:
    """Построить полосы, площадки перекрёстков и разметку (Шаг 2.3, п. 1 и 3)
    из `raw_roads` (с сохранённой связностью узлов,
    `osm.raw_roads.fetch_raw_roads_in_buffer`).

    Возвращает пустой `StreetNetwork`, если дорог нет вовсе (osm2streets
    ожидает непустую дорожную сеть, пустой набор — не ошибка, а факт
    отсутствия дорог в буфере)."""
    if not raw_roads:
        return StreetNetwork(lanes=[], intersections=[], markings=[])

    osm_xml = build_osm_xml(raw_roads)
    clip_geojson = _circle_geojson_wgs84(center_lon, center_lat, radius_m, zone)
    result = _run_osm2streets(osm_xml, clip_geojson, import_options or DEFAULT_IMPORT_OPTIONS)

    center_x, center_y, _ = wgs84_to_msk59(center_lon, center_lat, zone=zone)
    tags_by_way_id = {road.osm_id: road.tags for road in raw_roads}

    lanes = []
    for polygon, props in _localize_features(
        result.get("lanes", {}).get("features", []), zone, center_x, center_y, radius_m
    ):
        way_ids = tuple(props.get("osm_way_ids", []))
        lane_type = str(props.get("type", ""))
        lanes.append(
            LaneRibbon(
                osm_way_ids=way_ids,
                lane_type=lane_type,
                width_m=float(props.get("width", 0.0)),
                direction=str(props.get("direction", "")),
                polygon=polygon,
                surface=_surface_for_lane(lane_type, way_ids, tags_by_way_id),
            )
        )
    lanes += _build_curb_strips(lanes, tags_by_way_id)

    intersections = [
        IntersectionArea(kind=str(props.get("type", "")), polygon=polygon)
        for polygon, props in _localize_features(
            result.get("intersections", {}).get("features", []), zone, center_x, center_y, radius_m
        )
    ]

    markings = [
        LaneMarking(kind=str(props.get("type", "")), polygon=polygon)
        for polygon, props in _localize_features(
            result.get("markings", {}).get("features", []), zone, center_x, center_y, radius_m
        )
    ]

    return StreetNetwork(lanes=lanes, intersections=intersections, markings=markings)
