import { FormEvent, useState } from "react";
import { api, ChatAnswer, ChatTurn } from "../api";

type Entry = ChatTurn & { citations?: ChatAnswer["citations"] };

export default function ChatPanel({ onFocusCell }: { onFocusCell: (h3: string) => void }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(ev: FormEvent) {
    ev.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    const history = entries.map(({ role, content }) => ({ role, content }));
    setEntries([...entries, { role: "user", content: message }]);
    setInput("");
    setBusy(true);
    try {
      const a = await api.chat(message, history);
      setEntries((e) => [...e, { role: "assistant", content: a.answer, citations: a.citations }]);
    } catch (x) {
      setEntries((e) => [...e, { role: "assistant", content: `Erreur : ${String((x as Error).message ?? x)}` }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat">
      <h2>Assistant</h2>
      <div className="messages">
        {entries.length === 0 && <p className="hint">Ex. « Où ouvrir une pharmacie à Maârif ? » ou « Quelles règles pour ouvrir une pharmacie ? »</p>}
        {entries.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="content">{m.content}</div>
            {m.citations && (m.citations.model.length > 0 || m.citations.documents.length > 0) && (
              <div className="cites">
                {m.citations.model.slice(0, 8).map((c, j) => (
                  <button key={`m${j}`} className="cite model" onClick={() => onFocusCell(c.h3_id)}>
                    {c.rank ? `#${c.rank} ` : ""}{c.h3_id.slice(0, 9)}…
                  </button>
                ))}
                {m.citations.documents.map((d, j) => (
                  <span key={`d${j}`} className="cite doc">{d.document}{d.page ? `, p. ${d.page}` : ""}</span>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="msg assistant"><div className="content">…</div></div>}
      </div>
      <form onSubmit={send}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Posez une question…" />
        <button disabled={busy}>Envoyer</button>
      </form>
    </section>
  );
}
