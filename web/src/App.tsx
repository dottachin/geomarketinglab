import { useCallback, useEffect, useMemo, useState } from "react";
import { api, MapData, ModelInfo, Option, Poi } from "./api";
import Controls, { Constraints, DEFAULT_VIEW, METRICS, View } from "./components/Controls";
import MapView from "./components/MapView";
import CellPanel from "./components/CellPanel";
import ChatPanel from "./components/ChatPanel";
import TopList from "./components/TopList";
import { fmt, poiGroup } from "./lib/palette";
import { CellLabel, cellLabel } from "./lib/names";

const DEFAULT_CONSTRAINTS: Constraints = { maxCompetitors: null, excludeLowConfidence: false, opportunityOnly: false, minPopulation: null };

function loadView(): View {
  try {
    const v = JSON.parse(localStorage.getItem("geomlab.view") ?? "null");
    return v ? { ...DEFAULT_VIEW, ...v, poiGroups: { ...DEFAULT_VIEW.poiGroups, ...v.poiGroups } } : DEFAULT_VIEW;
  } catch { return DEFAULT_VIEW; }
}

export default function App() {
  const [cities, setCities] = useState<Option[]>([]);
  const [types, setTypes] = useState<Option[]>([]);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [city, setCity] = useState("casablanca");
  const [bt, setBt] = useState("pharmacy");
  const [constraints, setConstraints] = useState<Constraints>(DEFAULT_CONSTRAINTS);
  const [view, setView] = useState<View>(loadView);
  const [data, setData] = useState<MapData | null>(null);
  const [pois, setPois] = useState<Poi[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [highlights, setHighlights] = useState<string[]>([]);
  const [focus, setFocus] = useState<{ h3: string; n: number } | null>(null);
  const [tab, setTab] = useState<"rank" | "layers">("rank");
  const [chatOpen, setChatOpen] = useState(true);
  const [prompt, setPrompt] = useState<{ text: string; n: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.cities().then(setCities).catch((e) => setError(String(e.message ?? e)));
    api.businessTypes().then(setTypes).catch((e) => setError(String(e.message ?? e)));
    api.model().then(setModel).catch(() => setModel(null));
  }, []);

  useEffect(() => { try { localStorage.setItem("geomlab.view", JSON.stringify(view)); } catch { /* private mode */ } }, [view]);

  useEffect(() => {
    let live = true;
    setLoading(true); setSelected(null); setHighlights([]);
    api.map(city, bt)
      .then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e) => { if (live) { setData(null); setError(String(e.message ?? e)); } })
      .finally(() => live && setLoading(false));
    return () => { live = false; };
  }, [city, bt]);

  useEffect(() => {
    let live = true;
    setPois([]);
    api.pois(city).then((r) => live && setPois(r.pois)).catch(() => live && setPois([]));
    return () => { live = false; };
  }, [city]);

  const cells = useMemo(() => (data?.cells ?? []).filter((c) =>
    (constraints.maxCompetitors === null || c.competitors_catchment <= constraints.maxCompetitors) &&
    (constraints.minPopulation === null || (c.pop_catchment ?? 0) >= constraints.minPopulation) &&
    (!constraints.excludeLowConfidence || !c.low_confidence) &&
    (!constraints.opportunityOnly || c.opportunity)), [data, constraints]);

  // Colour by percentile over the whole city (not the filtered subset), so colours stay put when filtering.
  const colorT = useMemo(() => {
    const all = data?.cells ?? [];
    const sorted = [...all].sort((a, b) => a[view.metric] - b[view.metric]);
    const n = Math.max(1, sorted.length - 1);
    return new Map(sorted.map((c, i) => [c.h3_id, i / n]));
  }, [data, view.metric]);

  const visiblePois = useMemo(() => pois.filter((p) => {
    if (p.role === "business" && p.category === bt) return false; // drawn as competitors
    const g = poiGroup(p.role, p.category);
    return g ? view.poiGroups[g.id] : false;
  }), [pois, bt, view.poiGroups]);

  const poiCounts = useMemo(() => {
    const out: Record<string, number> = {};
    for (const p of pois) {
      if (p.role === "business" && p.category === bt) continue;
      const g = poiGroup(p.role, p.category);
      if (g) out[g.id] = (out[g.id] ?? 0) + 1;
    }
    return out;
  }, [pois, bt]);

  const focusCell = useCallback((h3: string) => {
    setSelected(h3);
    setFocus((f) => ({ h3, n: (f?.n ?? 0) + 1 }));
  }, []);
  const selectFromMap = useCallback((h3: string | null) => setSelected(h3), []);
  const ask = useCallback((text: string) => { setChatOpen(true); setPrompt((p) => ({ text, n: (p?.n ?? 0) + 1 })); }, []);

  const btLabel = types.find((t) => t.slug === bt)?.names?.en ?? bt;
  const cityLabel = cities.find((c) => c.slug === city)?.name ?? city;
  const opportunities = cells.filter((c) => c.opportunity).length;
  // Area names for every scored cell, so the chat never shows a raw H3 id.
  const totalCells = data?.cells.length ?? 0;
  const cellById = useMemo(() => new Map((data?.cells ?? []).map((c) => [c.h3_id, c])), [data]);
  const cellNames = useMemo(() => new Map<string, CellLabel>((data?.cells ?? []).map((c) =>
    [c.h3_id, cellLabel(c, c.rank, data?.cells.length ?? null, c.h3_id)])), [data]);

  return (
    <div className={`app ${chatOpen ? "chat-open" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            <svg viewBox="0 0 24 24" width="22" height="22"><path fill="currentColor" d="M12 2 20.66 7v10L12 22l-8.66-5V7L12 2Zm0 4.3L7.07 9.15v5.7L12 17.7l4.93-2.85v-5.7L12 6.3Z" /></svg>
          </span>
          <div><b>GeoMarketingLab</b><small>Where to open your business in Morocco</small></div>
        </div>
        <nav className="pickers">
          <div className="segmented city">
            {cities.map((c) => <button key={c.slug} className={c.slug === city ? "on" : ""} onClick={() => setCity(c.slug)}>{c.name}</button>)}
          </div>
          <div className="pills">
            {types.map((t) => <button key={t.slug} className={t.slug === bt ? "pill on" : "pill"} onClick={() => setBt(t.slug)}>{t.names?.en ?? t.slug}</button>)}
          </div>
        </nav>
        <div className="top-right">
          {model && <span className="badge" title={`Model ${model.model_version} · family ${model.family}`}>Model {model.family} · data as of {model.data_date}</span>}
          {!chatOpen && <button className="primary" onClick={() => setChatOpen(true)}>✦ Assistant</button>}
        </div>
      </header>

      <aside className="sidebar">
        <div className="stats-strip">
          <div><b>{fmt(cells.length)}</b><span>areas</span></div>
          <div><b>{fmt(opportunities)}</b><span>opportunities</span></div>
          <div><b>{fmt(data?.competitors.length ?? 0)}</b><span>competitors</span></div>
        </div>
        {error && <div className="banner error"><b>Data unavailable.</b> {error}</div>}
        {selected ? (
          <CellPanel h3={selected} bt={bt} btLabel={btLabel} cityLabel={cityLabel} cellById={cellById} totalCells={totalCells} onClose={() => setSelected(null)} onSelect={focusCell} onAsk={ask} />
        ) : (
          <>
            <div className="tabs">
              <button className={tab === "rank" ? "on" : ""} onClick={() => setTab("rank")}>Best locations</button>
              <button className={tab === "layers" ? "on" : ""} onClick={() => setTab("layers")}>Filters & layers</button>
            </div>
            {tab === "rank"
              ? (loading ? <div className="list-skel">{Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton" />)}</div>
                : <TopList cells={cells} total={totalCells} colorT={colorT} selected={selected} onSelect={focusCell} onHover={setHovered} />)
              : <Controls constraints={constraints} onConstraints={setConstraints} view={view} onView={setView}
                  btLabel={btLabel} poiCounts={poiCounts} competitorCount={data?.competitors.length ?? 0} />}
          </>
        )}
        <p className="attribution">
          © OpenStreetMap (ODbL) · HCP RGPH 2024 · Copernicus, VIIRS, WorldPop via Google Earth Engine.
          Scores rank how closely each area resembles places where this business already exists, adjusted for competition. They are not a sales forecast.
        </p>
      </aside>

      <main className="map">
        <MapView city={city} cells={cells} total={totalCells} colorT={colorT} metricLabel={METRICS[view.metric].label}
          competitors={data?.competitors ?? []} pois={visiblePois} showCompetitors={view.showCompetitors}
          opacity={view.opacity} extruded={view.extruded} basemap={view.basemap}
          selected={selected} hovered={hovered} highlights={highlights} focus={focus} onSelect={selectFromMap} />
        {loading && <div className="map-loading"><span className="spinner" /> Loading scores…</div>}
        <div className="map-legend">
          <b>{METRICS[view.metric].label}</b>
          <div className="ramp" />
          <div className="ramp-labels"><span>{METRICS[view.metric].low}</span><span>{METRICS[view.metric].high}</span></div>
          {highlights.length > 0 && (
            <button className="link" onClick={() => setHighlights([])}>Clear the {highlights.length} places cited by the assistant</button>
          )}
        </div>
      </main>

      {chatOpen && (
        <ChatPanel city={city} cityLabel={cityLabel} bt={bt} btLabel={btLabel} selected={selected} prompt={prompt} cellNames={cellNames}
          onCell={focusCell} onHighlights={setHighlights} onClose={() => setChatOpen(false)} />
      )}
    </div>
  );
}
