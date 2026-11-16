import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { api, ChatEvent, ChatTurn, Citations } from "../api";
import Markdown from "../lib/Markdown";
import type { CellLabel } from "../lib/names";

type Step = { name: string; done: boolean; error: boolean };
type Entry = ChatTurn & { citations?: Citations; steps?: Step[]; error?: boolean; seconds?: number; model?: string };

const STEP_LABEL: Record<string, string> = {
  recommend_locations: "Ranking locations",
  explain_location: "Analysing an area",
  search_documents: "Searching documents",
  query_stats: "Computing statistics",
  simulate_location: "Simulating a scenario",
};

type Props = {
  city: string; cityLabel: string; bt: string; btLabel: string; selected: string | null; cellNames: ReadonlyMap<string, CellLabel>;
  prompt: { text: string; n: number } | null;
  onCell: (h3: string) => void; onHighlights: (h3s: string[]) => void; onClose: () => void;
};

export default function ChatPanel({ city, cityLabel, bt, btLabel, selected, cellNames, prompt, onCell, onHighlights, onClose }: Props) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<{ text: string; steps: Step[] }>({ text: "", steps: [] });
  const [elapsed, setElapsed] = useState(0);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [entries, live]);

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [busy]);

  useEffect(() => { if (prompt) void send(prompt.text); }, [prompt]); // eslint-disable-line react-hooks/exhaustive-deps

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    const history = entries.filter((e) => !e.error).map(({ role, content }) => ({ role, content })).slice(-10);
    setEntries((e) => [...e, { role: "user", content: message }]);
    setInput("");
    setBusy(true);
    setElapsed(0);
    setLive({ text: "", steps: [] });
    const t0 = Date.now();
    let steps: Step[] = [];
    let finished = false;
    const ctl = new AbortController();
    abort.current = ctl;
    const handle = (ev: ChatEvent) => {
      if (ev.type === "tool") {
        steps = [...steps, { name: ev.name, done: false, error: false }];
        setLive({ text: "", steps });
      } else if (ev.type === "tool_result") {
        const i = steps.map((s) => s.name).lastIndexOf(ev.name);
        if (i >= 0) steps = steps.map((s, j) => (j === i ? { ...s, done: true, error: ev.is_error } : s));
        setLive((l) => ({ ...l, steps }));
      } else if (ev.type === "delta") {
        setLive((l) => ({ ...l, text: l.text + ev.text }));
      } else if (ev.type === "reset") {
        setLive((l) => ({ ...l, text: "" }));
      } else if (ev.type === "done") {
        finished = true;
        const cited = [...new Set(ev.citations.model.map((c) => c.h3_id))];
        if (cited.length) onHighlights(cited);
        setEntries((e) => [...e, { role: "assistant", content: ev.answer || "(empty answer)", citations: ev.citations,
          steps, seconds: Math.round((Date.now() - t0) / 1000), model: ev.model }]);
      } else if (ev.type === "error") {
        finished = true;
        setEntries((e) => [...e, { role: "assistant", content: ev.detail, error: true, steps }]);
      }
    };
    try {
      await api.chatStream(message, history, { city, business_type: bt, h3_id: selected }, handle, ctl.signal);
      if (!finished) setEntries((e) => [...e, { role: "assistant", content: "The connection was interrupted before the answer was complete.", error: true, steps }]);
    } catch (x) {
      const aborted = (x as Error).name === "AbortError";
      setEntries((e) => [...e, { role: "assistant", error: true, steps,
        content: aborted ? "Answer stopped." : `Error: ${String((x as Error).message ?? x)}` }]);
    } finally {
      setBusy(false);
      setLive({ text: "", steps: [] });
      abort.current = null;
    }
  }

  const submit = (ev: FormEvent) => { ev.preventDefault(); void send(input); };
  const onKey = (ev: KeyboardEvent<HTMLTextAreaElement>) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); void send(input); }
  };

  const btLower = btLabel.toLowerCase();
  const selectedName = selected ? cellNames.get(selected)?.chip ?? "the selected area" : null;
  const suggestions = [
    `Top 5 locations for a ${btLower} in ${cityLabel}`,
    `Where are the competitor-free opportunities for a ${btLower} in ${cityLabel}?`,
    ...(selected ? [`Why does \`${selected}\` get this score?`] : []),
    `What are the rules for opening a ${btLower} in Morocco?`,
  ];

  return (
    <section className="chat">
      <header className="chat-head">
        <div>
          <h2>Assistant</h2>
          <p className="subtle">Answers grounded in the model and the document corpus</p>
        </div>
        <div className="chat-actions">
          {entries.length > 0 && !busy && <button className="ghost" onClick={() => { setEntries([]); onHighlights([]); }}>Clear</button>}
          <button className="icon" onClick={onClose} aria-label="Close the assistant">×</button>
        </div>
      </header>

      <div className="messages" ref={scroller}>
        {entries.length === 0 && !busy && (
          <div className="welcome">
            <div className="welcome-mark">✦</div>
            <p>Ask about the best locations, a neighbourhood or the regulations, in English, French or Arabic. The map context ({cityLabel}, {btLabel}{selectedName ? `, ${selectedName}` : ""}) is sent automatically.</p>
            <div className="suggestions">
              {suggestions.map((s) => (
                <button key={s} onClick={() => void send(s)}>
                  <Markdown text={s} onCell={() => undefined} names={cellNames} inline />
                </button>
              ))}
            </div>
          </div>
        )}
        {entries.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.error ? " error" : ""}`}>
            {m.steps && m.steps.length > 0 && <Steps steps={m.steps} />}
            <div className="bubble" dir="auto">
              {m.error ? m.content : <Markdown text={m.content} onCell={onCell} names={cellNames} />}
            </div>
            {m.citations && (m.citations.model.length > 0 || m.citations.documents.length > 0) && (
              <div className="cites">
                {m.citations.model.length > 0 && (
                  <button className="cite all" onClick={() => onHighlights([...new Set(m.citations!.model.map((c) => c.h3_id))])}>
                    ◎ Show the {new Set(m.citations.model.map((c) => c.h3_id)).size} places on the map
                  </button>
                )}
                {m.citations.model.slice(0, 10).map((c, j) => (
                  <button key={`m${j}`} className="cite model" dir="auto" onClick={() => onCell(c.h3_id)} title={cellNames.get(c.h3_id)?.full ?? c.h3_id}>
                    {cellNames.get(c.h3_id)?.chip ?? (c.rank ? `Area ranked #${c.rank}` : "Area on the map")}
                  </button>
                ))}
                {m.citations.documents.map((d, j) => (
                  <span key={`d${j}`} className="cite doc" title={d.section ?? ""}>📄 {d.document}{d.page ? `, p. ${d.page}` : ""}</span>
                ))}
              </div>
            )}
            {m.seconds !== undefined && <div className="msg-meta">{m.model} · {m.seconds} s</div>}
          </div>
        ))}
        {busy && (
          <div className="msg assistant">
            <Steps steps={live.steps} pending />
            <div className="bubble" dir="auto">
              {live.text ? <Markdown text={live.text} onCell={onCell} names={cellNames} /> : <span className="typing"><i /><i /><i /></span>}
            </div>
            <div className="msg-meta">
              {elapsed} s{elapsed > 20 && " · a local model on CPU can take 1 to 3 minutes"}
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea dir="auto" aria-label="Message to the assistant" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} rows={2}
          placeholder="Ask a question… (Enter to send)" disabled={busy} />
        {busy
          ? <button type="button" className="stop" onClick={() => abort.current?.abort()}>Stop</button>
          : <button type="submit" disabled={!input.trim()}>Send</button>}
      </form>
    </section>
  );
}

function Steps({ steps, pending = false }: { steps: Step[]; pending?: boolean }) {
  const shown = pending && steps.length === 0 ? [{ name: "_thinking", done: false, error: false }] : steps;
  return (
    <ul className="steps">
      {shown.map((s, i) => (
        <li key={i} className={s.error ? "err" : s.done ? "done" : "run"}>
          <span className="st-icon">{s.error ? "!" : s.done ? "✓" : ""}</span>
          {s.name === "_thinking" ? "The model is thinking…" : STEP_LABEL[s.name] ?? s.name}
        </li>
      ))}
      {pending && steps.length > 0 && steps.every((s) => s.done) && (
        <li className="run"><span className="st-icon" />Writing the answer…</li>
      )}
    </ul>
  );
}
