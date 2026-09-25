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

local srid = 4326

local function def_table(name, geom_type)
    return osm2pgsql.define_table({
        name = name,
        ids = { type = 'any', id_column = 'osm_id', type_column = 'osm_type' },
        columns = {
            { column = 'tags', type = 'jsonb' },
            { column = 'geom', type = geom_type, projection = srid, not_null = true },
        },
    })
end

local tables = {
    buildings = def_table('osm_buildings', 'geometry'),       -- полигоны/мультиполигоны
    roads = def_table('osm_roads', 'linestring'),
    railways = def_table('osm_railways', 'linestring'),
    water_areas = def_table('osm_water_areas', 'geometry'),    -- полигоны/мультиполигоны
    waterways = def_table('osm_waterways', 'linestring'),
    vegetation = def_table('osm_vegetation', 'geometry'),      -- точки (дерево) + полигоны (лес/газон)
    power = def_table('osm_power', 'geometry'),                -- точки (опоры) + линии (провода) + полигоны (подстанции)
    landscaping = def_table('osm_landscaping', 'geometry'),    -- скамейки, фонари, ограждения, площадки
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

    if tags.building then
        insert_way_geom(tables.buildings, object, true)
        return
    end

    if tags.highway then
        insert_way_geom(tables.roads, object, false)
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
