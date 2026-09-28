"""Расчётные зоны ограничений (Шаг 3.2) — правила расчёта в этом (отдельном
от `store.py`) файле, каждое со ссылкой на норму (п. 1); зона считается
только там, где нет официальной (`zones_needing_calculation`, п. 2); статус
`"расчётно"` — уже часть модели данных `constraints.store` (Шаг 3.1, п. 3),
здесь не переопределяется.

Ссылка «конфигурационный файл» реализована как модуль с именованными
таблицами/константами (тот же приём, что уже даёт проект — например,
`SAFETY_ZONE_HALF_WIDTH_M_BY_VOLTAGE_KV` в `geometry/power.py`, откуда
здесь и переиспользуется, не дублируется), а не отдельный YAML/JSON — в
кодовой базе нет YAML-конфигов нигде ещё, вводить новый парсер/зависимость
ради этого не оправдано.
"""

from __future__ import annotations

from topology_geo.geometry.power import SAFETY_ZONE_HALF_WIDTH_M_BY_VOLTAGE_KV, _safety_zone_half_width_m

# --- СЗЗ кладбищ по площади ---------------------------------------------
# СанПиН 2.2.1/2.1.1.1200-03, разд. VII («Кладбища»): санитарно-защитная
# зона зависит от площади кладбища. Верхняя граница диапазона включительно.
CEMETERY_SZZ_M_BY_AREA_HA: list[tuple[float, float]] = [
    (10.0, 50.0),
    (20.0, 100.0),
    (40.0, 300.0),
]


def calculate_cemetery_szz_m(area_ha: float) -> float:
    """СЗЗ кладбища по площади (СанПиН 2.2.1/2.1.1.1200-03). Площадь свыше
    верхней границы таблицы (40 га) — берётся максимальное табличное
    значение, честно (не экстраполируется)."""
    for upper_ha, szz_m in CEMETERY_SZZ_M_BY_AREA_HA:
        if area_ha <= upper_ha:
            return szz_m
    return CEMETERY_SZZ_M_BY_AREA_HA[-1][1]


# --- водоохранные зоны по длине реки ------------------------------------
# Водный кодекс РФ, ст. 65, ч. 4: ширина водоохранной зоны реки/ручья по
# длине от истока. Верхняя граница диапазона включительно.
WATER_PROTECTION_ZONE_M_BY_LENGTH_KM: list[tuple[float, float]] = [
    (10.0, 50.0),
    (50.0, 100.0),
]
WATER_PROTECTION_ZONE_M_OVER_50KM = 200.0


def calculate_water_protection_zone_m(river_length_km: float) -> float:
    """Ширина водоохранной зоны (Водный кодекс РФ, ст. 65, ч. 4)."""
    for upper_km, zone_m in WATER_PROTECTION_ZONE_M_BY_LENGTH_KM:
        if river_length_km <= upper_km:
            return zone_m
    return WATER_PROTECTION_ZONE_M_OVER_50KM


# --- прибрежная защитная полоса по уклону берега ------------------------
# Водный кодекс РФ, ст. 65, ч. 11: ширина прибрежной защитной полосы
# зависит от уклона берега, НЕ от длины реки (отдельная норма от
# водоохранной зоны выше). Обратный/нулевой уклон — 30 м, до 3° — 40 м,
# 3° и более — 50 м.
COASTAL_STRIP_DEFAULT_M = 30.0  # уклон неизвестен - минимальное (не завышающее) значение, честно


def calculate_coastal_protective_strip_m(bank_slope_deg: float | None) -> float:
    """Ширина прибрежной защитной полосы (Водный кодекс РФ, ст. 65, ч. 11).
    Без известного уклона берега — честный минимум (30 м), не выдуманное
    среднее."""
    if bank_slope_deg is None:
        return COASTAL_STRIP_DEFAULT_M
    if bank_slope_deg <= 0:
        return 30.0
    if bank_slope_deg < 3:
        return 40.0
    return 50.0


def calculate_power_line_zone_half_width_m(voltage_kv: float | None) -> float | None:
    """Охранная зона ЛЭП по напряжению — переиспользует ТУ ЖЕ таблицу и
    функцию, что и геометрия ЛЭП Шага 2.7, п. 5 (`geometry.power`,
    ПП РФ №160 от 24.02.2009), а не дублирует значения."""
    return _safety_zone_half_width_m(voltage_kv)


# --- охранные зоны прочих сетей по типу и диаметру ----------------------
# Упрощённая, представительная таблица (не исчерпывающая): охранная зона
# инженерных сетей регулируется РАЗНЫМИ документами по типу сети
# (водопровод/канализация — СП 42.13330.2016; газораспределительные сети —
# ПП РФ №878 от 20.11.2000; тепловые сети — СП 124.13330.2012) — здесь
# сведены минимальные охранные расстояния по типу сети, диаметр повышает
# класс, где это предусмотрено нормой (газ высокого давления, диаметр
# ≥300 мм — увеличенная зона по ПП РФ №878).
NETWORK_TYPE_LABELS = {
    "К": "канализация", "В": "водопровод", "Т": "теплосеть", "Г": "газопровод", "Кл": "кабельная линия",
}

_NETWORK_ZONE_M_BY_TYPE_AND_DIAMETER: dict[str, list[tuple[float, float]]] = {
    # (верхняя граница диаметра, мм включительно) -> охранная зона, м
    "В": [(300.0, 5.0), (1_000_000.0, 10.0)],  # СП 42.13330.2016
    "К": [(300.0, 5.0), (1_000_000.0, 10.0)],  # СП 42.13330.2016
    "Т": [(1_000_000.0, 3.0)],  # СП 124.13330.2012 (охранная зона теплотрассы)
    "Г": [(300.0, 2.0), (1_000_000.0, 7.0)],  # ПП РФ №878 (газ низкого/высокого давления, упрощённо по диаметру)
    "Кл": [(1_000_000.0, 1.0)],  # ПУЭ, кабельные линии до 35 кВ
}


def calculate_network_protection_zone_m(network_type: str, diameter_mm: float) -> float:
    """Охранная зона инженерной сети по типу (К/В/Т/Г/Кл) и диаметру.
    Неизвестный тип сети — честная ошибка, не молчаливое умолчание."""
    table = _NETWORK_ZONE_M_BY_TYPE_AND_DIAMETER.get(network_type)
    if table is None:
        raise ValueError(f"неизвестный тип сети: {network_type!r} (ожидается один из {list(NETWORK_TYPE_LABELS)})")
    for upper_mm, zone_m in table:
        if diameter_mm <= upper_mm:
            return zone_m
    return table[-1][1]


def zones_needing_calculation(conn, bbox: tuple[float, float, float, float], zone_type: str) -> bool:
    """Действие п. 2: расчёт зоны только там, где официальной зоны ЭТОГО
    вида ещё нет. `True` — считать нужно (официальной зоны нет или её
    покрытие не полное - упрощённо: нет НИ ОДНОЙ официальной зоны этого
    вида, пересекающей bbox)."""
    from topology_geo.constraints.store import STATUS_OFFICIAL, find_zones

    existing = find_zones(conn, bbox)
    return not any(z.zone_type == zone_type and z.status == STATUS_OFFICIAL for z in existing)
