import { Option } from "../api";

export type Constraints = { maxCompetitors: number | null; excludeLowConfidence: boolean; opportunityOnly: boolean };

type Props = {
  cities: Option[]; types: Option[]; city: string; bt: string; constraints: Constraints;
  onCity: (c: string) => void; onBt: (b: string) => void; onConstraints: (c: Constraints) => void;
};

export default function Controls({ cities, types, city, bt, constraints, onCity, onBt, onConstraints }: Props) {
  return (
    <div className="controls">
      <label>Ville
        <select value={city} onChange={(e) => onCity(e.target.value)}>
          {cities.map((c) => <option key={c.slug} value={c.slug}>{c.name}</option>)}
        </select>
      </label>
      <label>Activité
        <select value={bt} onChange={(e) => onBt(e.target.value)}>
          {types.map((t) => <option key={t.slug} value={t.slug}>{t.names?.fr ?? t.slug}</option>)}
        </select>
      </label>
      <label>Concurrents max. dans l'hexagone
        <input type="number" min={0} value={constraints.maxCompetitors ?? ""} placeholder="—"
          onChange={(e) => onConstraints({ ...constraints, maxCompetitors: e.target.value === "" ? null : Number(e.target.value) })} />
      </label>
      <label className="check">
        <input type="checkbox" checked={constraints.opportunityOnly}
          onChange={(e) => onConstraints({ ...constraints, opportunityOnly: e.target.checked })} />
        Opportunités seulement (forte adéquation, aucun concurrent)
      </label>
      <label className="check">
        <input type="checkbox" checked={constraints.excludeLowConfidence}
          onChange={(e) => onConstraints({ ...constraints, excludeLowConfidence: e.target.checked })} />
        Masquer les cellules à faible confiance
      </label>
    </div>
  );
}
