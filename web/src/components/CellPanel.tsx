import { useEffect, useState } from "react";
import { api, Explanation } from "../api";

const fmt = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? "—" : v.toLocaleString("fr-FR", { maximumFractionDigits: d }));

export default function CellPanel({ h3, bt, onClose }: { h3: string; bt: string; onClose: () => void }) {
  const [e, setE] = useState<Explanation | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setE(null);
    api.explain(h3, bt).then(setE).catch((x) => setErr(String(x.message ?? x)));
  }, [h3, bt]);

  if (err) return <div className="panel"><button onClick={onClose}>×</button><p className="error">{err}</p></div>;
  if (!e) return <div className="panel">Chargement…</div>;
  const maxAbs = Math.max(...e.drivers.map((d) => Math.abs(d.shap)), 1e-6);
  return (
    <div className="panel">
      <div className="panel-head">
        <strong>{e.district ?? "Hexagone"}</strong>
        <button onClick={onClose} aria-label="Fermer">×</button>
      </div>
      <code className="h3">{e.h3_id}</code>
      <div className="kpis">
        <div><span>Rang</span><b>#{e.city_rank}</b></div>
        <div><span>Score final</span><b>{e.final_score.toFixed(3)}</b></div>
        <div><span>Adéquation</span><b>{e.suitability.toFixed(3)}</b></div>
        <div><span>Saturation</span><b>{e.saturation.toFixed(2)}</b></div>
      </div>
      {e.low_confidence && <p className="warn">Faible confiance : {e.confidence_reasons.join(", ")}</p>}
      <h3>Pourquoi ce score (SHAP)</h3>
      <ul className="drivers">
        {e.drivers.map((d) => (
          <li key={d.feature} title={`${d.feature} — ${d.family}`}>
            <span className="lbl">{d.label}</span>
            <span className={`bar ${d.shap > 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(d.shap) / maxAbs) * 100}%` }} />
            <span className="val">{fmt(d.value, 2)} {d.unit}</span>
          </li>
        ))}
      </ul>
      <h3>Concurrents à moins de {e.catchment_m} m ({e.competitors_in_catchment})</h3>
      <ul className="plain">
        {e.nearby_competitors.length === 0 && <li>Aucun</li>}
        {e.nearby_competitors.map((c, i) => <li key={i}>{c.name ?? "Sans nom"} — {fmt(c.distance_m)} m</li>)}
      </ul>
      <p className="meta">Population dans la zone : {fmt(e.population_in_catchment)} · modèle <code>{e.model_version}</code> · données {e.data_date}</p>
    </div>
  );
}
