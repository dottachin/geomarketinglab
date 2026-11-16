import { MapCell } from "../api";
import { fmt, ramp, css } from "../lib/palette";
import { placeSubtitle, placeTitle, rankLabel } from "../lib/names";

type Props = {
  cells: MapCell[]; total: number; colorT: Map<string, number>; selected: string | null;
  onSelect: (h3: string) => void; onHover: (h3: string | null) => void;
};

export default function TopList({ cells, total, colorT, selected, onSelect, onHover }: Props) {
  const top = cells.slice(0, 30);
  const max = Math.max(...top.map((c) => c.final_score), 1e-6);
  if (top.length === 0) return <p className="empty">No area matches the filters.</p>;
  return (
    <ol className="toplist" onMouseLeave={() => onHover(null)}>
      {top.map((c) => (
        <li key={c.h3_id}>
          <button className={c.h3_id === selected ? "row on" : "row"} onClick={() => onSelect(c.h3_id)} onMouseEnter={() => onHover(c.h3_id)}>
            <span className="rank" style={{ background: css(ramp(colorT.get(c.h3_id) ?? 0)) }}>{c.rank}</span>
            <span className="row-main">
              <span className="row-title" dir="auto" title={`${rankLabel(c.rank, total)} · ${c.h3_id}`}>{placeTitle(c)}</span>
              {placeSubtitle(c) && <span className="row-place" dir="auto">{placeSubtitle(c)}</span>}
              <span className="row-sub">
                {c.competitors_catchment} competitor{c.competitors_catchment === 1 ? "" : "s"} · {fmt(c.pop_catchment)} residents
                {c.opportunity && <em className="tag ok">opportunity</em>}
                {c.low_confidence && <em className="tag warn">low confidence</em>}
              </span>
              <span className="scorebar"><span style={{ width: `${(c.final_score / max) * 100}%` }} /></span>
            </span>
            <span className="row-score">{c.final_score.toFixed(3)}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
