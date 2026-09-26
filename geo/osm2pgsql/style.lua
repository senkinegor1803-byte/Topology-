-- Стиль osm2pgsql (flex output) для импорта OSM в PostGIS (Шаг 1.1 плана).
--
-- Раскладывает объекты по отдельным таблицам (здания, дороги, ж/д, вода,
-- растительность, электросети, благоустройство), как требует Шаг 1.1.
-- Теги НЕ разбираются на колонки здесь: весь набор тегов объекта целиком
-- кладётся в jsonb-колонку `tags` — приведение к словарю данных (Шаг 0.8:
-- этажность, покрытие, напряжение и т.д.) выполняется позже, на выборке
-- участка (Шаг 1.4, `topology_geo`), а не на импорте всего края. Так менять
-- правила соответствия тегов не нужно перезаливать всю базу OSM.
--
-- Мультиполигон-отношения (крупные здания и водоёмы с "дырками") обработаны
-- отдельно; известный краевой случай — если внешний way такого отношения сам
-- по себе несёт тот же тег (например `building=yes`), он может попасть в
-- таблицу и как отдельный way, и как часть отношения. Это принятый в
-- официальных примерах osm2pgsql компромисс для MVP; при необходимости
-- решается дедупликацией по osm_id на выборке (Шаг 1.4).
--
-- `building:part=*` (составные здания, Шаг 2.2, п. 1) — отдельная таблица
-- `osm_building_parts`, тем же способом, что и `osm_buildings`; по конвенции
-- OSM (wiki: Key:building:part) это разные теги на разных объектах, way с
-- `building:part` в этом стиле не проверяется на `building` (см.
-- `process_way`). `entrance=*` (Шаг 2.2, п. 3) — точки, `osm_entrances`.
--
-- `osm_roads.nodes` (Шаг 2.3, п. 1) — доп. колонка с массивом ID узлов way
-- (`object.nodes`, тот же порядок, что и вершины `geom`) поверх обычного
-- набора tags/geom. Нужна, чтобы позже (`topology_geo.osm.raw_roads`) при
-- построении полос через osm2streets восстановить настоящий граф узлов
-- (общий ID узла на перекрёстке = топологическая связь), не имея доступа к
-- исходному .osm/.pbf повторно — osm2pgsql его не хранит, а разложенные по
-- отдельным объектам геометрия/теги в PostGIS сами по себе связность не
-- сохраняют.

local srid = 4326

local function def_table(name, geom_type, extra_columns)
    local columns = {
        { column = 'tags', type = 'jsonb' },
        { column = 'geom', type = geom_type, projection = srid, not_null = true },
    }
    for _, c in ipairs(extra_columns or {}) do
        columns[#columns + 1] = c
    end
    return osm2pgsql.define_table({
        name = name,
        ids = { type = 'any', id_column = 'osm_id', type_column = 'osm_type' },
        columns = columns,
    })
end

local tables = {
    buildings = def_table('osm_buildings', 'geometry'),       -- полигоны/мультиполигоны
    building_parts = def_table('osm_building_parts', 'geometry'), -- building:part=*, полигоны (Шаг 2.2, п. 1)
    roads = def_table('osm_roads', 'linestring', { { column = 'nodes', type = 'jsonb' } }),
    railways = def_table('osm_railways', 'linestring'),
    water_areas = def_table('osm_water_areas', 'geometry'),    -- полигоны/мультиполигоны
    waterways = def_table('osm_waterways', 'linestring'),
    vegetation = def_table('osm_vegetation', 'geometry'),      -- точки (дерево) + полигоны (лес/газон)
    power = def_table('osm_power', 'geometry'),                -- точки (опоры) + линии (провода) + полигоны (подстанции)
    landscaping = def_table('osm_landscaping', 'geometry'),    -- скамейки, фонари, ограждения, площадки
    entrances = def_table('osm_entrances', 'point'),            -- entrance=*, точки (Шаг 2.2, п. 3)
}

local function has_any(tags, keys)
    for _, k in ipairs(keys) do
        if tags[k] then
            return true
        end
    end
    return false
end

local function insert_way_geom(table_ref, object, as_area)
    if as_area then
        if object.is_closed then
            table_ref:insert({ tags = object.tags, geom = object:as_polygon() })
        end
    else
        table_ref:insert({ tags = object.tags, geom = object:as_linestring() })
    end
end

function osm2pgsql.process_node(object)
    local tags = object.tags

    if tags.entrance then
        tables.entrances:insert({ tags = tags, geom = object:as_point() })
        return
    end

    if tags.natural == 'tree' then
        tables.vegetation:insert({ tags = tags, geom = object:as_point() })
        return
    end

    if tags.power == 'tower' or tags.power == 'pole' or tags.power == 'substation' then
        tables.power:insert({ tags = tags, geom = object:as_point() })
        return
    end

    if has_any(tags, { 'amenity', 'leisure', 'barrier', 'highway' })
        and (tags.amenity == 'bench' or tags.amenity == 'waste_basket'
             or tags.leisure == 'playground'
             or tags.highway == 'street_lamp'
             or tags.barrier)
    then
        tables.landscaping:insert({ tags = tags, geom = object:as_point() })
    end
end

function osm2pgsql.process_way(object)
    local tags = object.tags

    if tags['building:part'] then
        insert_way_geom(tables.building_parts, object, true)
        return
    end

    if tags.building then
        insert_way_geom(tables.buildings, object, true)
        return
    end

    if tags.highway then
        tables.roads:insert({ tags = tags, nodes = object.nodes, geom = object:as_linestring() })
        return
    end

    if tags.railway then
        insert_way_geom(tables.railways, object, false)
        return
    end

    if tags.waterway then
        insert_way_geom(tables.waterways, object, false)
        return
    end

    if tags.natural == 'water' or tags.landuse == 'reservoir' then
        insert_way_geom(tables.water_areas, object, true)
        return
    end

    if tags.natural == 'wood' or tags.landuse == 'forest' or tags.landuse == 'grass' then
        insert_way_geom(tables.vegetation, object, true)
        return
    end

    if tags.power == 'line' or tags.power == 'minor_line' then
        insert_way_geom(tables.power, object, false)
        return
    end
    if tags.power == 'substation' or tags.power == 'plant' then
        insert_way_geom(tables.power, object, true)
        return
    end

    if tags.leisure == 'park' or tags.leisure == 'playground'
        or tags.barrier == 'fence' or tags.barrier == 'wall'
    then
        insert_way_geom(tables.landscaping, object, tags.barrier == nil)
    end
end

function osm2pgsql.select_relation_members(relation)
    if relation.tags.type == 'multipolygon' then
        return { ways = osm2pgsql.way_member_ids(relation) }
    end
end

function osm2pgsql.process_relation(object)
    local tags = object.tags
    if tags.type ~= 'multipolygon' then
        return
    end

    if tags.building then
        tables.buildings:insert({ tags = tags, geom = object:as_multipolygon() })
    elseif tags.natural == 'water' or tags.landuse == 'reservoir' then
        tables.water_areas:insert({ tags = tags, geom = object:as_multipolygon() })
    elseif tags.natural == 'wood' or tags.landuse == 'forest' then
        tables.vegetation:insert({ tags = tags, geom = object:as_multipolygon() })
    end
end
