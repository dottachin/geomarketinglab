import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { PickingInfo } from "@deck.gl/core";
import { cellToLatLng } from "h3-js";
import { Competitor, MapCell, Poi } from "../api";
import { placeSubtitle, placeTitle, rankLabel } from "../lib/names";
import { CATEGORY_LABEL, COMPETITOR, fmt, poiGroup, ramp } from "../lib/palette";

export const BASEMAPS = {
  voyager: { label: "Map", url: "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json" },
  light: { label: "Light", url: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json" },
  dark: { label: "Dark", url: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" },
} as const;
export type Basemap = keyof typeof BASEMAPS;

const CENTERS: Record<string, [number, number]> = {
  casablanca: [-7.59, 33.57], rabat: [-6.84, 34.0], marrakech: [-8.0, 31.63],
};

type Props = {
  city: string; cells: MapCell[]; total: number; colorT: Map<string, number>; metricLabel: string;
  competitors: Competitor[]; pois: Poi[]; showCompetitors: boolean;
  opacity: number; extruded: boolean; basemap: Basemap;
  selected: string | null; hovered: string | null; highlights: string[];
  focus: { h3: string; n: number } | null;
  onSelect: (h3: string | null) => void;
};

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function firstLabelLayer(map: maplibregl.Map): string | undefined {
  return map.getStyle().layers?.find((l) => l.type === "symbol")?.id;
}

export default function MapView(p: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const overlay = useRef<MapboxOverlay | null>(null);
  const fittedCity = useRef<string | null>(null);
  const [labelId, setLabelId] = useState<string | undefined>();
  const [ready, setReady] = useState(false);

  // create the map once
  useEffect(() => {
    if (!container.current) return;
    const m = new maplibregl.Map({
      container: container.current, style: BASEMAPS[p.basemap].url,
      center: CENTERS[p.city] ?? CENTERS.casablanca, zoom: 11.5, attributionControl: { compact: true },
    });
    overlay.current = new MapboxOverlay({ interleaved: true, layers: [] });
    m.addControl(overlay.current as unknown as maplibregl.IControl);
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    m.on("style.load", () => { setLabelId(firstLabelLayer(m)); setReady(true); });
    map.current = m;
    return () => { m.remove(); map.current = null; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // basemap switch
  const currentStyle = useRef<string>(BASEMAPS[p.basemap].url);
  useEffect(() => {
    const url = BASEMAPS[p.basemap].url;
    if (map.current && currentStyle.current !== url) {
      currentStyle.current = url;
      map.current.setStyle(url);
    }
  }, [p.basemap]);

  // 3D tilt
  useEffect(() => {
    map.current?.easeTo({ pitch: p.extruded ? 50 : 0, bearing: p.extruded ? -15 : 0, duration: 800 });
  }, [p.extruded]);

  // fit the city's cells once per city
  useEffect(() => {
    const m = map.current;
    if (!m || p.cells.length === 0 || fittedCity.current === p.city) return;
    fittedCity.current = p.city;
    let [w, s, e, n] = [180, 90, -180, -90];
    for (const c of p.cells) {
      const [lat, lon] = cellToLatLng(c.h3_id);
      w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat);
    }
    m.fitBounds([[w, s], [e, n]], { padding: 40, duration: 1200 });
  }, [p.city, p.cells]);

  // fly to a cell chosen in the list, the chat or the panel
  useEffect(() => {
    const m = map.current;
    if (!m || !p.focus) return;
    const [lat, lon] = cellToLatLng(p.focus.h3);
    m.flyTo({ center: [lon, lat], zoom: Math.max(m.getZoom(), 14.2), duration: 1100 });
  }, [p.focus]);

  // deck.gl layers
  useEffect(() => {
    if (!overlay.current || !ready) return;
    const alpha = Math.round(p.opacity * 255);
    const sel = p.selected, hov = p.hovered;
    const hl = new Set(p.highlights);
    // Draw the hexagons below the basemap's street/place labels (read by MapboxOverlay, untyped in deck 9.0).
    const underLabels = (p.extruded ? {} : { beforeId: labelId }) as Record<string, never>;
    const outlined = p.cells.filter((c) => c.h3_id === sel || c.h3_id === hov || hl.has(c.h3_id));
    overlay.current.setProps({
      layers: [
        new H3HexagonLayer<MapCell>({
          id: "scores", data: p.cells, pickable: true, extruded: p.extruded, stroked: !p.extruded, filled: true,
          ...underLabels,
          getHexagon: (d) => d.h3_id,
          getFillColor: (d) => [...ramp(p.colorT.get(d.h3_id) ?? 0), alpha],
          getElevation: (d) => (p.colorT.get(d.h3_id) ?? 0) ** 2 * 900,
          elevationScale: 1, material: { ambient: 0.6, diffuse: 0.6, shininess: 20 },
          getLineColor: [255, 255, 255, 50], getLineWidth: 1, lineWidthUnits: "pixels",
          autoHighlight: true, highlightColor: [255, 255, 255, 110],
          updateTriggers: { getFillColor: [p.colorT, alpha], getElevation: p.colorT },
          transitions: { getFillColor: 400, getElevation: 600 },
          onClick: (info) => { if (info.object) p.onSelect(info.object.h3_id); },
        }),
        new H3HexagonLayer<MapCell>({
          id: "outlines", data: outlined, pickable: false, filled: false, stroked: true, extruded: false,
          getHexagon: (d) => d.h3_id, lineWidthUnits: "pixels",
          getLineColor: (d) => (d.h3_id === sel ? [17, 24, 39, 255] : d.h3_id === hov ? [255, 255, 255, 255] : [79, 70, 229, 255]),
          getLineWidth: (d) => (d.h3_id === sel ? 4 : 3),
          updateTriggers: { getLineColor: [sel, hov], getLineWidth: sel },
        }),
        new ScatterplotLayer<Poi>({
          id: "pois", data: p.pois, pickable: true, radiusUnits: "meters", getRadius: 18,
          radiusMinPixels: 2, radiusMaxPixels: 6, stroked: true, lineWidthMinPixels: 0.5,
          getLineColor: [255, 255, 255, 200],
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: (d) => [...(poiGroup(d.role, d.category)?.color ?? [120, 120, 120]), 210],
        }),
        new ScatterplotLayer<Competitor>({
          id: "competitors", data: p.competitors, visible: p.showCompetitors, pickable: true,
          radiusUnits: "meters", getRadius: 30, radiusMinPixels: 3.5, radiusMaxPixels: 9,
          stroked: true, lineWidthMinPixels: 1.2, getLineColor: [255, 255, 255, 255],
          getPosition: (d) => [d.lon, d.lat], getFillColor: [...COMPETITOR, 235],
        }),
      ],
      getTooltip: (info: PickingInfo) => {
        const o = info.object;
        if (!o) return null;
        let html: string;
        if (info.layer?.id === "scores") {
          const c = o as MapCell;
          html = `<div class="tt-title" dir="auto">${esc(placeTitle(c))}</div>
            ${placeSubtitle(c) ? `<div class="tt-sub" dir="auto">${esc(placeSubtitle(c))}</div>` : ""}
            <div class="tt-row"><span>Rank</span><b>${esc(rankLabel(c.rank, p.total))}</b></div>
            <div class="tt-row"><span>Final score</span><b>${c.final_score.toFixed(3)}</b></div>
            <div class="tt-row"><span>Suitability</span><b>${c.suitability.toFixed(3)}</b></div>
            <div class="tt-row"><span>Saturation</span><b>${c.saturation.toFixed(2)}</b></div>
            <div class="tt-row"><span>Competitors (catchment)</span><b>${c.competitors_catchment}</b></div>
            <div class="tt-row"><span>Population (catchment)</span><b>${fmt(c.pop_catchment)}</b></div>
            ${c.opportunity ? '<div class="tt-tag ok">Opportunity</div>' : ""}
            ${c.low_confidence ? '<div class="tt-tag warn">Low confidence</div>' : ""}
            <div class="tt-hint">Click for details · colour: ${esc(p.metricLabel)}</div>`;
        } else if (info.layer?.id === "competitors") {
          html = `<div class="tt-title" dir="auto">${esc((o as Competitor).name ?? "Unnamed")}</div><div class="tt-hint">Existing competitor</div>`;
        } else {
          const poi = o as Poi;
          html = `<div class="tt-title" dir="auto">${esc(poi.name ?? "Unnamed")}</div><div class="tt-hint">${esc(CATEGORY_LABEL[poi.category] ?? poi.category)}</div>`;
        }
        return { html, className: "map-tooltip" };
      },
    });
  }, [p.cells, p.colorT, p.competitors, p.pois, p.showCompetitors, p.opacity, p.extruded, p.selected, p.hovered,
      p.highlights, p.metricLabel, p.total, labelId, ready, p.onSelect]);

  return <div ref={container} className="map-canvas" />;
}
