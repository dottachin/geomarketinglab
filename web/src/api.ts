// Thin client for the FastAPI backend (proxied at /api).
export type Option = { slug: string; name?: string; names?: Record<string, string> };

export type MapCell = {
  h3_id: string; final_score: number; suitability: number; saturation: number;
  rank: number; n_existing: number; opportunity: boolean; low_confidence: boolean;
};
export type MapData = {
  city: string; business_type: string; model_version: string; data_date: string;
  cells: MapCell[]; competitors: { name: string | null; lon: number; lat: number }[];
};
export type Driver = { feature: string; label: string; family: string; unit: string; value: number | null; shap: number; effect: string };
export type Explanation = {
  h3_id: string; district: string | null; final_score: number; suitability: number; saturation: number;
  city_rank: number; percentile: number; opportunity: boolean; existing_in_cell: number;
  competitors_in_catchment: number; catchment_m: number; population_in_catchment: number | null;
  low_confidence: boolean; confidence_reasons: string[]; drivers: Driver[];
  key_statistics: { feature: string; label: string; value: number; unit: string; source: string }[];
  nearby_competitors: { name: string | null; distance_m: number }[];
  model_version: string; data_date: string;
};
export type ChatTurn = { role: "user" | "assistant"; content: string };
export type ChatAnswer = {
  answer: string;
  citations: {
    model: { h3_id: string; business_type: string; rank?: number }[];
    documents: { document: string; page: number | null; section: string | null; similarity: number }[];
    model_versions: { model_version: string; data_date: string }[];
  };
  tool_calls: { name: string; args: Record<string, unknown>; is_error: boolean }[];
};

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  cities: () => get<Option[]>("/meta/cities"),
  businessTypes: () => get<Option[]>("/meta/business-types"),
  map: (city: string, bt: string) => get<MapData>(`/map/${city}/${bt}`),
  explain: (h3: string, bt: string) => get<Explanation>(`/cells/${h3}/explain?business_type=${bt}`),
  chat: async (message: string, history: ChatTurn[]): Promise<ChatAnswer> => {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `${r.status}`);
    return r.json();
  },
};
