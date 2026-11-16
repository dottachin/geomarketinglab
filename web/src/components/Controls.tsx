import type { ReactNode } from "react";
import { POI_GROUPS, RAMP_CSS, css, COMPETITOR } from "../lib/palette";
import { BASEMAPS, Basemap } from "./MapView";

export type Metric = "final_score" | "suitability" | "saturation";
export const METRICS: Record<Metric, { label: string; low: string; high: string }> = {
  final_score: { label: "Final score", low: "low", high: "high" },
  suitability: { label: "Suitability (ignoring competition)", low: "low", high: "high" },
  saturation: { label: "Market saturation", low: "open", high: "saturated" },
};
export type Constraints = { maxCompetitors: number | null; excludeLowConfidence: boolean; opportunityOnly: boolean; minPopulation: number | null };
export type View = {
  metric: Metric; opacity: number; extruded: boolean; basemap: Basemap;
  showCompetitors: boolean; poiGroups: Record<string, boolean>;
};
export const DEFAULT_VIEW: View = {
  metric: "final_score", opacity: 0.72, extruded: false, basemap: "voyager", showCompetitors: true,
  poiGroups: Object.fromEntries(POI_GROUPS.map((g) => [g.id, g.id === "transport" || g.id === "business"])),
};

type Props = {
  constraints: Constraints; onConstraints: (c: Constraints) => void;
  view: View; onView: (v: View) => void; btLabel: string; poiCounts: Record<string, number>; competitorCount: number;
};

function Toggle({ on, onChange, children }: { on: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" aria-hidden />
      <span className="toggle-label">{children}</span>
    </label>
  );
}

export default function Controls({ constraints, onConstraints, view, onView, btLabel, poiCounts, competitorCount }: Props) {
  const set = (patch: Partial<View>) => onView({ ...view, ...patch });
  return (
    <div className="controls">
      <section>
        <h3>Area colour</h3>
        <div className="segmented">
          {(Object.keys(METRICS) as Metric[]).map((m) => (
            <button key={m} className={view.metric === m ? "on" : ""} onClick={() => set({ metric: m })}>
              {m === "final_score" ? "Score" : m === "suitability" ? "Suitability" : "Saturation"}
            </button>
          ))}
        </div>
        <div className="legend-ramp"><div className="ramp" style={{ background: RAMP_CSS }} />
          <div className="ramp-labels"><span>{METRICS[view.metric].low}</span><span>{METRICS[view.metric].high}</span></div>
        </div>
        <label className="slider">Opacity
          <input type="range" min={0.15} max={1} step={0.05} value={view.opacity} onChange={(e) => set({ opacity: Number(e.target.value) })} />
        </label>
        <Toggle on={view.extruded} onChange={(v) => set({ extruded: v })}>3D view (height = score)</Toggle>
      </section>

      <section>
        <h3>Filters</h3>
        <label className="field">Max. competitors in catchment
          <input type="number" min={0} value={constraints.maxCompetitors ?? ""} placeholder="No limit"
            onChange={(e) => onConstraints({ ...constraints, maxCompetitors: e.target.value === "" ? null : Number(e.target.value) })} />
        </label>
        <label className="field">Min. catchment population
          <input type="number" min={0} step={1000} value={constraints.minPopulation ?? ""} placeholder="No limit"
            onChange={(e) => onConstraints({ ...constraints, minPopulation: e.target.value === "" ? null : Number(e.target.value) })} />
        </label>
        <Toggle on={constraints.opportunityOnly} onChange={(v) => onConstraints({ ...constraints, opportunityOnly: v })}>
          Opportunities only <small>high suitability, no competitor</small>
        </Toggle>
        <Toggle on={constraints.excludeLowConfidence} onChange={(v) => onConstraints({ ...constraints, excludeLowConfidence: v })}>
          Hide low-confidence areas
        </Toggle>
      </section>

      <section>
        <h3>Places on the map</h3>
        <Toggle on={view.showCompetitors} onChange={(v) => set({ showCompetitors: v })}>
          <i className="swatch" style={{ background: css(COMPETITOR) }} />Existing {btLabel.toLowerCase()} sites <small>{competitorCount}</small>
        </Toggle>
        {POI_GROUPS.map((g) => (
          <Toggle key={g.id} on={view.poiGroups[g.id]} onChange={(v) => set({ poiGroups: { ...view.poiGroups, [g.id]: v } })}>
            <i className="swatch" style={{ background: css(g.color) }} />{g.label} <small>{poiCounts[g.id] ?? 0}</small>
          </Toggle>
        ))}
      </section>

      <section>
        <h3>Basemap</h3>
        <div className="segmented">
          {(Object.keys(BASEMAPS) as Basemap[]).map((b) => (
            <button key={b} className={view.basemap === b ? "on" : ""} onClick={() => set({ basemap: b })}>{BASEMAPS[b].label}</button>
          ))}
        </div>
      </section>
    </div>
  );
}
