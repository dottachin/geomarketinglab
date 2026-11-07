import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import { MapCell, MapData } from "../api";

const CENTERS: Record<string, [number, number]> = {
  casablanca: [-7.59, 33.57], rabat: [-6.84, 34.0], marrakech: [-8.0, 31.63],
};
const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

// Sequential ramp (light → dark) for final_score; scores are probabilities in [0, 1].
function color(score: number, maxScore: number): [number, number, number, number] {
  const t = Math.min(1, score / Math.max(maxScore, 1e-6));
  const stops = [[255, 245, 235], [253, 174, 107], [241, 105, 19], [166, 54, 3], [94, 22, 2]];
  const x = t * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + f * (stops[i + 1][k] - v)));
  return [c[0], c[1], c[2], 190];
}

type Props = {
  city: string; cells: MapCell[]; competitors: MapData["competitors"];
  selected: string | null; onSelect: (h3: string) => void;
};

export default function MapView({ city, cells, competitors, selected, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const overlay = useRef<MapboxOverlay | null>(null);

  useEffect(() => {
    if (!container.current) return;
    map.current = new maplibregl.Map({ container: container.current, style: STYLE, center: CENTERS[city] ?? [-7.59, 33.57], zoom: 11 });
    overlay.current = new MapboxOverlay({ interleaved: false, layers: [] });
    map.current.addControl(overlay.current as unknown as maplibregl.IControl);
    map.current.addControl(new maplibregl.NavigationControl(), "top-right");
    return () => map.current?.remove();
  }, []);

  useEffect(() => {
    map.current?.flyTo({ center: CENTERS[city] ?? [-7.59, 33.57], zoom: 11 });
  }, [city]);

  useEffect(() => {
    const maxScore = cells.reduce((m, c) => Math.max(m, c.final_score), 0);
    overlay.current?.setProps({
      layers: [
        new H3HexagonLayer<MapCell>({
          id: "scores", data: cells, pickable: true, extruded: false, stroked: true,
          getHexagon: (d) => d.h3_id,
          getFillColor: (d) => color(d.final_score, maxScore),
          getLineColor: (d) => (d.h3_id === selected ? [20, 20, 20, 255] : [255, 255, 255, 60]),
          getLineWidth: (d) => (d.h3_id === selected ? 3 : 1), lineWidthUnits: "pixels",
          updateTriggers: { getLineColor: selected, getLineWidth: selected, getFillColor: maxScore },
          onClick: (info) => info.object && onSelect(info.object.h3_id),
        }),
        new ScatterplotLayer({
          id: "competitors", data: competitors, getPosition: (d: { lon: number; lat: number }) => [d.lon, d.lat],
          getRadius: 4, radiusUnits: "pixels", getFillColor: [37, 99, 235, 220], stroked: true, getLineColor: [255, 255, 255],
        }),
      ],
      getTooltip: ({ object }: { object?: MapCell }) =>
        object && object.h3_id
          ? `#${object.rank} · score ${object.final_score.toFixed(3)}${object.low_confidence ? " · faible confiance" : ""}`
          : null,
    });
  }, [cells, competitors, selected, onSelect]);

  return (
    <div className="map-wrap">
      <div ref={container} className="map-canvas" />
      <div className="legend">
        <span>score final</span><div className="ramp" /><span className="dot" /> concurrents existants
      </div>
    </div>
  );
}
