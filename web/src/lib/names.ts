// Human-readable names for hexagons: users never see raw H3 ids.

const ARRONDISSEMENT_RE = /^arrondissement\s+(?:(?:de\s+l|d)['’]\s*|(?:de\s+la|des|du|de)\s+)?/i;

/** "Arrondissement d'Aïn-Chock" -> "Aïn-Chock"; names without the prefix are returned unchanged. */
export function areaName(district: string | null | undefined): string | null {
  const s = district?.trim();
  if (!s) return null;
  const name = s.replace(ARRONDISSEMENT_RE, "").trim() || s;
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/** Naming fields shared by MapCell, Explanation and neighbouring cells (all optional: older APIs lack some). */
export type Place = { neighbourhood?: string | null; landmark?: string | null; district?: string | null };

/** Main name of a cell: neighbourhood, else the district's short name. Never contains the rank. */
export function placeTitle(p: Place): string {
  return p.neighbourhood?.trim() || areaName(p.district) || "Unnamed area";
}

/** "near Jemaa el-Fna · Medina" (landmark and/or district; the district is omitted when it is the title). */
export function placeSubtitle(p: Place): string {
  const title = placeTitle(p);
  const district = areaName(p.district);
  const parts = [
    p.landmark?.trim() ? `near ${p.landmark.trim()}` : null,
    district && district.toLowerCase() !== title.toLowerCase() ? district : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

/** Short chip text for the chat: "Bab Doukkala (near Koutoubia)". */
export function chipLabel(p: Place): string {
  const lm = p.landmark?.trim();
  return lm ? `${placeTitle(p)} (near ${lm})` : placeTitle(p);
}

/** "#2 of 1,888" (or "#2" when the total is unknown). */
export function rankLabel(rank: number, total?: number | null): string {
  return total ? `#${rank.toLocaleString("en-US")} of ${total.toLocaleString("en-US")}` : `#${rank.toLocaleString("en-US")}`;
}

/** Chat label of a cell: short chip text plus a full description for the tooltip. */
export type CellLabel = { chip: string; full: string };
export function cellLabel(p: Place, rank: number, total: number | null, h3: string): CellLabel {
  const sub = placeSubtitle(p);
  return { chip: chipLabel(p), full: `${placeTitle(p)}${sub ? ` (${sub})` : ""} · ${rankLabel(rank, total)} · ${h3}` };
}

const REASONS: Record<string, string> = {
  sparse_osm_coverage: "few mapped places nearby",
  missing_satellite_data: "no satellite data",
  missing_census_data: "no census data",
  few_training_examples: "few training examples",
};

/** Readable text for a confidence_reasons code from the API. */
export const reasonLabel = (code: string) => REASONS[code] ?? code.replace(/_/g, " ");
