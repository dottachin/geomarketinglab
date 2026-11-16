import { useEffect, useState } from "react";
import { api, Explanation, MapCell } from "../api";
import { fmt } from "../lib/palette";
import { placeSubtitle, placeTitle, rankLabel, reasonLabel } from "../lib/names";

type Props = {
  h3: string; bt: string; btLabel: string; cityLabel: string; cellById: ReadonlyMap<string, MapCell>; totalCells: number;
  onClose: () => void; onSelect: (h3: string) => void; onAsk: (question: string) => void;
};

export default function CellPanel({ h3, bt, btLabel, cityLabel, cellById, totalCells, onClose, onSelect, onAsk }: Props) {
  const [e, setE] = useState<Explanation | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setE(null); setErr(null);
    api.explain(h3, bt).then((x) => live && setE(x)).catch((x) => live && setErr(String(x.message ?? x)));
    return () => { live = false; };
  }, [h3, bt]);

  const head = (
    <div className="detail-head">
      <button className="back" onClick={onClose}>← Ranking</button>
    </div>
  );
  if (err) return <div className="detail">{head}<p className="error">{err}</p></div>;
  if (!e) return <div className="detail">{head}<div className="skeleton" /><div className="skeleton short" /><div className="skeleton" /></div>;

  const neighbourTitle = (n: Explanation["neighbouring_cells"][number]) => {
    const c = cellById.get(n.h3_id);
    return placeTitle({ neighbourhood: n.neighbourhood ?? c?.neighbourhood, district: c?.district });
  };
  const maxAbs = Math.max(...e.drivers.map((d) => Math.abs(d.shap)), 1e-6);
  const pct = Math.round(e.percentile * 100);
  const total = e.n_cells ?? (totalCells || null);
  const sub = placeSubtitle(e);
  return (
    <div className="detail">
      {head}
      <h2 dir="auto" title={e.district ?? undefined}>{placeTitle(e)}</h2>
      {sub && <p className="place-sub" dir="auto">{sub}</p>}
      <p className="subtle">{btLabel} · <b>{rankLabel(e.city_rank, total)}</b> in {cityLabel} · better than {pct}% of the city's areas</p>
      <div className="pct"><span style={{ width: `${pct}%` }} /></div>

      <div className="kpis">
        <div><span>Final score</span><b>{e.final_score.toFixed(3)}</b></div>
        <div><span>Suitability</span><b>{e.suitability.toFixed(3)}</b></div>
        <div><span>Saturation</span><b>{e.saturation.toFixed(2)}</b></div>
        <div><span>Competitors ≤ {e.catchment_m} m</span><b>{e.competitors_in_catchment}</b></div>
        <div><span>In this area</span><b>{e.existing_in_cell}</b></div>
        <div><span>Catchment population</span><b>{fmt(e.population_in_catchment)}</b></div>
      </div>
      {e.opportunity && <p className="note ok">Opportunity: high suitability and no competitor in this area.</p>}
      {e.low_confidence && <p className="note warn">Low confidence: {e.confidence_reasons.length ? e.confidence_reasons.map(reasonLabel).join(", ") : "limited data"}</p>}

      <button className="ask" onClick={() => onAsk(`Why does \`${e.h3_id}\` rank #${e.city_rank} in ${cityLabel} for a ${btLabel.toLowerCase()}? What are its strengths and risks?`)}>
        ✦ Ask the assistant for an analysis
      </button>

      <h3>What drives the score <small>SHAP contributions</small></h3>
      <ul className="drivers">
        {e.drivers.map((d) => (
          <li key={d.feature} title={`${d.feature} — ${d.family}`}>
            <span className="lbl">{d.label}</span>
            <span className="val">{fmt(d.value, 2)} {d.unit}</span>
            <span className="diverge">
              <span className={d.shap > 0 ? "bar pos" : "bar neg"} style={{ width: `${(Math.abs(d.shap) / maxAbs) * 50}%` }} />
            </span>
          </li>
        ))}
      </ul>
      <p className="legend-inline"><i className="dotc neg" /> lowers <i className="dotc pos" /> raises</p>

      {e.key_statistics.length > 0 && <>
        <h3>Key statistics</h3>
        <div className="stats">
          {e.key_statistics.map((s) => (
            <div key={s.feature} title={s.source}><span>{s.label}</span><b>{fmt(s.value, 2)} <small>{s.unit}</small></b></div>
          ))}
        </div>
      </>}

      <h3>Competitors within {e.catchment_m} m</h3>
      {e.nearby_competitors.length === 0 ? <p className="subtle">None</p> : (
        <ul className="plain">
          {e.nearby_competitors.map((c, i) => <li key={i}><span dir="auto">{c.name ?? "Unnamed"}</span><b>{fmt(c.distance_m)} m</b></li>)}
        </ul>
      )}

      {e.neighbouring_cells.length > 0 && <>
        <h3>Neighbouring areas</h3>
        <div className="neighbours">
          {e.neighbouring_cells.map((n) => (
            <button key={n.h3_id} onClick={() => onSelect(n.h3_id)} title={n.h3_id}
              aria-label={`${neighbourTitle(n)}, rank ${n.city_rank}, score ${Number(n.final_score).toFixed(3)}`}>
              <span dir="auto">{neighbourTitle(n)}</span><small>#{n.city_rank} · {Number(n.final_score).toFixed(3)}</small>
            </button>
          ))}
        </div>
      </>}
      <p className="meta">Model <code>{e.model_version}</code> · data as of {e.data_date} · <code className="h3" title="H3 cell id">{e.h3_id}</code></p>
    </div>
  );
}
