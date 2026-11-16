// Minimal, safe Markdown for chat messages (no HTML injection): headings, lists, tables,
// **bold**, *italic*, `code`. H3 cell ids become chips showing the area name that focus the map.
// Every block gets dir="auto" so Arabic / Darija paragraphs render right-to-left.
import { Fragment, ReactNode } from "react";
import type { CellLabel } from "./names";

const H3_RE = /\b(8[0-9a-f]{14})\b/g;
const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;

type Names = ReadonlyMap<string, CellLabel>;
type Ctx = { onCell: (h3: string) => void; names: Names; plain?: boolean };

function withCells(text: string, { onCell, names, plain }: Ctx, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(H3_RE)) {
    const i = m.index ?? 0;
    if (i > last) out.push(text.slice(last, i));
    const n = names.get(m[1]);
    const label = n?.chip ?? "Area on the map";
    const full = n?.full ?? m[1];
    out.push(plain
      ? <span key={`${key}-${i}`} className="cell-chip" dir="auto" title={full}>{label}</span>
      : <button key={`${key}-${i}`} className="cell-chip" dir="auto" onClick={() => onCell(m[1])} title={`Show on the map: ${full}`}>{label}</button>);
    last = i + m[1].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function inline(text: string, ctx: Ctx, key: string): ReactNode[] {
  return text.split(INLINE_RE).filter(Boolean).map((part, i) => {
    const k = `${key}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={k}>{withCells(part.slice(2, -2), ctx, k)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) {
      const code = part.slice(1, -1);
      return /^8[0-9a-f]{14}$/.test(code) ? <Fragment key={k}>{withCells(code, ctx, k)}</Fragment> : <code key={k}>{code}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={k}>{withCells(part.slice(1, -1), ctx, k)}</em>;
    return <Fragment key={k}>{withCells(part, ctx, k)}</Fragment>;
  });
}

const NO_NAMES: Names = new Map();

type Props = {
  text: string; onCell: (h3: string) => void; names?: Names;
  /** Render one line of inline markup inside a <span> (chips are not buttons), e.g. inside another button. */
  inline?: boolean;
};

export default function Markdown({ text, onCell, names = NO_NAMES, inline: asInline = false }: Props) {
  const ctx: Ctx = { onCell, names, plain: asInline };
  if (asInline) return <span dir="auto">{inline(text, ctx, "i")}</span>;
  const lines = text.replace(/\r/g, "").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const k = `b${i}`;
    if (!line.trim()) { i++; continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { blocks.push(<h4 key={k} dir="auto">{inline(h[2], ctx, k)}</h4>); i++; continue; }
    if (/^\s*\|/.test(line)) {
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) rows.push(cells);
        i++;
      }
      const [head, ...body] = rows;
      blocks.push(
        <div key={k} className="md-table"><table dir="auto">
          {head && <thead><tr>{head.map((c, j) => <th key={j}>{inline(c, ctx, `${k}h${j}`)}</th>)}</tr></thead>}
          <tbody>{body.map((r, ri) => <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c, ctx, `${k}${ri}${j}`)}</td>)}</tr>)}</tbody>
        </table></div>);
      continue;
    }
    const listRe = /^\s*([-*•]|\d+[.)])\s+(.*)$/;
    if (listRe.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items: string[] = [];
      while (i < lines.length && listRe.test(lines[i])) { items.push(lines[i].match(listRe)![2]); i++; }
      const Tag = ordered ? "ol" : "ul";
      blocks.push(<Tag key={k}>{items.map((it, j) => <li key={j} dir="auto">{inline(it, ctx, `${k}-${j}`)}</li>)}</Tag>);
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !listRe.test(lines[i]) && !/^(#{1,4})\s/.test(lines[i]) && !/^\s*\|/.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    blocks.push(<p key={k} dir="auto">{inline(para.join(" "), ctx, k)}</p>);
  }
  return <div className="md">{blocks}</div>;
}
