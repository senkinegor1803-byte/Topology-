// Обёртка над osm2streets-js (Шаг 2.3, п. 1): получить геометрию полос
// (проезжая часть/тротуар/парковка/...) и перекрёстков из OSM XML с
// сохранённой топологией узлов (см. topology_geo.osm.raw_roads.build_osm_xml
// и docstring geo/src/topology_geo/geometry/streets.py).
//
// Запуск: node convert.mjs <osm.xml> <clip.geojson> <options.json>
// Печатает в stdout один JSON-объект {lanes, intersections} (FeatureCollection
// каждый) или {error: "..."} с ненулевым кодом выхода при сбое.

import init, { JsStreetNetwork } from "osm2streets-js/osm2streets_js.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const [, , osmXmlPath, clipGeojsonPath, optionsPath] = process.argv;

if (!osmXmlPath || !clipGeojsonPath || !optionsPath) {
  console.error("usage: node convert.mjs <osm.xml> <clip.geojson> <options.json>");
  process.exit(2);
}

const wasmPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "node_modules/osm2streets-js/osm2streets_js_bg.wasm",
);

// osm2streets-js печатает отладочные строки (например "Found N nodes...")
// через console.info/console.log (в Node оба по умолчанию пишут в stdout) -
// перенаправляем всё, кроме console.error, в stderr, чтобы единственным,
// что попадает в stdout, был итоговый JSON (иначе вызывающий Python-код
// получит невалидный JSON с примесью логов).
for (const method of ["log", "info", "debug", "warn"]) {
  console[method] = (...args) => console.error(...args);
}

async function main() {
  await init(readFileSync(wasmPath));

  const osmXml = readFileSync(osmXmlPath, "utf8");
  const clipGeojson = readFileSync(clipGeojsonPath, "utf8");
  const options = JSON.parse(readFileSync(optionsPath, "utf8"));

  const network = new JsStreetNetwork(osmXml, clipGeojson, options);
  const result = {
    lanes: JSON.parse(network.toLanePolygonsGeojson()),
    intersections: JSON.parse(network.toIntersectionMarkingsGeojson()),
  };
  process.stdout.write(JSON.stringify(result));
}

main().catch((err) => {
  console.error(JSON.stringify({ error: String(err && err.message ? err.message : err) }));
  process.exit(1);
});
