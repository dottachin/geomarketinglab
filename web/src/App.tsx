import { useEffect, useMemo, useState } from "react";
import { api, MapData, Option } from "./api";
import Controls, { Constraints } from "./components/Controls";
import MapView from "./components/MapView";
import CellPanel from "./components/CellPanel";
import ChatPanel from "./components/ChatPanel";

const DEFAULT_CONSTRAINTS: Constraints = { maxCompetitors: null, excludeLowConfidence: false, opportunityOnly: false };

export default function App() {
  const [cities, setCities] = useState<Option[]>([]);
  const [types, setTypes] = useState<Option[]>([]);
  const [city, setCity] = useState("casablanca");
  const [bt, setBt] = useState("pharmacy");
  const [constraints, setConstraints] = useState<Constraints>(DEFAULT_CONSTRAINTS);
  const [data, setData] = useState<MapData | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.cities().then(setCities).catch((e) => setError(String(e)));
    api.businessTypes().then(setTypes).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    setData(null);
    setSelected(null);
    api.map(city, bt).then((d) => { setData(d); setError(null); }).catch((e) => setError(String(e.message ?? e)));
  }, [city, bt]);

  const cells = useMemo(() => {
    if (!data) return [];
    return data.cells.filter((c) =>
      (constraints.maxCompetitors === null || c.n_existing <= constraints.maxCompetitors) &&
      (!constraints.excludeLowConfidence || !c.low_confidence) &&
      (!constraints.opportunityOnly || c.opportunity));
  }, [data, constraints]);

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>GeoMarketingLab</h1>
        <Controls cities={cities} types={types} city={city} bt={bt} constraints={constraints}
          onCity={setCity} onBt={setBt} onConstraints={setConstraints} />
        {error && <p className="error">{error}</p>}
        {data && (
          <p className="meta">
            {cells.length} hexagons · model <code>{data.model_version}</code> · data {data.data_date}
          </p>
        )}
        {selected && <CellPanel h3={selected} bt={bt} onClose={() => setSelected(null)} />}
        <p className="attribution">
          Map data © OpenStreetMap contributors (ODbL) · HCP RGPH 2024 · Copernicus Sentinel-2, VIIRS, WorldPop via
          Google Earth Engine. Scores rank similarity to where businesses already operate — not a revenue forecast.
        </p>
      </aside>
      <main className="map">
        <MapView city={city} cells={cells} competitors={data?.competitors ?? []} selected={selected} onSelect={setSelected} />
      </main>
      <ChatPanel onFocusCell={setSelected} />
    </div>
  );
}
