// Thin client for the FastAPI backend (proxied at /api).
export type Option = { slug: string; name?: string; names?: Record<string, string> };

export type MapCell = {
  h3_id: string; district: string | null; neighbourhood?: string | null; landmark?: string | null;
  final_score: number; suitability: number; saturation: number;
  rank: number; n_existing: number; competitors_catchment: number; pop_catchment: number | null;
  opportunity: boolean; low_confidence: boolean;
};
export type Competitor = { name: string | null; lon: number; lat: number };
export type MapData = {
  city: string; business_type: string; model_version: string; data_date: string;
  cells: MapCell[]; competitors: Competitor[];
};
export type Poi = { name: string | null; role: "business" | "attractor"; category: string; lon: number; lat: number };
export type Driver = { feature: string; label: string; family: string; unit: string; value: number | null; shap: number; effect: string };
export type Explanation = {
  h3_id: string; district: string | null; neighbourhood?: string | null; landmark?: string | null;
  final_score: number; suitability: number; saturation: number;
  city_rank: number; n_cells?: number | null; percentile: number; opportunity: boolean; existing_in_cell: number;
  competitors_in_catchment: number; catchment_m: number; population_in_catchment: number | null;
  low_confidence: boolean; confidence_reasons: string[]; drivers: Driver[];
  key_statistics: { feature: string; label: string; value: number; unit: string; source: string }[];
  nearby_competitors: { name: string | null; distance_m: number }[];
  neighbouring_cells: { h3_id: string; final_score: number; city_rank: number; n_existing: number; neighbourhood?: string | null }[];
  model_version: string; data_date: string;
};
export type ModelInfo = { model_version: string; family: string; data_date: string; metrics: Record<string, number> };
export type ChatTurn = { role: "user" | "assistant"; content: string };
export type Citations = {
  model: { h3_id: string; business_type: string; rank?: number }[];
  documents: { document: string; page: number | null; section: string | null; similarity: number }[];
  model_versions: { model_version: string; data_date: string }[];
};
export type ChatAnswer = {
  answer: string; citations: Citations; model: string; stop_reason: string;
  tool_calls: { name: string; args: Record<string, unknown>; is_error: boolean }[];
};
export type ChatContext = { city?: string; business_type?: string; h3_id?: string | null };
export type ChatEvent =
  | { type: "tool"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; is_error: boolean }
  | { type: "delta"; text: string }
  | { type: "reset" }
  | { type: "ping" }
  | ({ type: "done" } & ChatAnswer)
  | { type: "error"; detail: string };

async function errorText(r: Response): Promise<string> {
  const body = await r.json().catch(() => ({}));
  const detail = body.error ?? body.detail;
  return typeof detail === "string" ? detail : `${r.status} ${r.statusText}`;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error(await errorText(r));
  return r.json();
}

/** POST /chat/stream and call onEvent for every NDJSON line until done/error. */
async function chatStream(message: string, history: ChatTurn[], context: ChatContext,
                          onEvent: (e: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch("/api/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, context }), signal,
  });
  if (!r.ok || !r.body) throw new Error(await errorText(r));
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as ChatEvent);
    }
  }
}

export const api = {
  cities: () => get<Option[]>("/meta/cities"),
  businessTypes: () => get<Option[]>("/meta/business-types"),
  model: () => get<ModelInfo>("/meta/model"),
  map: (city: string, bt: string) => get<MapData>(`/map/${city}/${bt}`),
  pois: (city: string) => get<{ city: string; pois: Poi[] }>(`/map/${city}/pois`),
  explain: (h3: string, bt: string) => get<Explanation>(`/cells/${h3}/explain?business_type=${bt}`),
  chatStream,
};
