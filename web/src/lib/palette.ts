// Colour helpers shared by the map, the legend and the panels.
export type RGB = [number, number, number];

// Viridis-like sequential ramp: readable on light and dark basemaps and for colour-blind users.
export const RAMP: RGB[] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
export const RAMP_CSS = `linear-gradient(90deg, ${RAMP.map((c) => `rgb(${c.join(",")})`).join(", ")})`;

export function ramp(t: number): RGB {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2);
  const f = x - i;
  return RAMP[i].map((v, k) => Math.round(v + f * (RAMP[i + 1][k] - v))) as RGB;
}
export const css = (c: RGB, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

export const COMPETITOR: RGB = [225, 29, 72];

// Points of interest grouped for the layer toggles (categories from config/poi_categories.yaml).
export type PoiGroup = { id: string; label: string; color: RGB; categories: string[] };
export const POI_GROUPS: PoiGroup[] = [
  { id: "business", label: "Other studied businesses", color: [234, 88, 12], categories: [] },
  { id: "transport", label: "Transport", color: [37, 99, 235], categories: ["bus_stop", "tram_stop", "train_station", "parking"] },
  { id: "shops", label: "Shops & services", color: [147, 51, 234], categories: ["retail_other", "restaurant", "bank", "market", "mall", "office"] },
  { id: "education", label: "Education", color: [5, 150, 105], categories: ["school", "university"] },
  { id: "health", label: "Health", color: [220, 38, 38], categories: ["clinic", "hospital"] },
  { id: "worship", label: "Places of worship", color: [161, 98, 7], categories: ["mosque"] },
];
export function poiGroup(role: string, category: string): PoiGroup | undefined {
  if (role === "business") return POI_GROUPS[0];
  return POI_GROUPS.find((g) => g.categories.includes(category)) ?? POI_GROUPS[2];
}

export const CATEGORY_LABEL: Record<string, string> = {
  pharmacy: "Pharmacy", cafe: "Cafe", bakery: "Bakery", gym: "Gym", supermarket: "Supermarket",
  bus_stop: "Bus stop", tram_stop: "Tram stop", train_station: "Train station", parking: "Parking",
  retail_other: "Shop", restaurant: "Restaurant", bank: "Bank", market: "Market", mall: "Shopping mall",
  office: "Offices", school: "School", university: "University", clinic: "Clinic", hospital: "Hospital", mosque: "Mosque",
};

export const fmt = (v: number | null | undefined, d = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: d });
