# RAG document corpus

Documents the assistant can cite through `search_documents` (doc §7.1 "Document knowledge").
Indexed by `geomlab index-docs` / the `index_documents` DAG (only changed files are re-embedded).

```
corpus/
  methodology/   how scores are built (shipped with the project)
  hcp/           HCP reports and RGPH publications (PDF)          ← add yours
  regulations/   commune regulations, licensing procedures (PDF)  ← add yours
  urbanism/      urban planning documents (plans d'aménagement)   ← add yours
```

Supported: `.pdf`, `.md`, `.txt`. Optional sidecar `<file>.meta.yaml`:

```yaml
title: "Procédure d'ouverture d'une pharmacie"
city: casablanca        # omit when national
year: 2024
lang: fr
source_url: https://…
```

Only add documents you are allowed to redistribute for academic use, and keep their
source URL in the sidecar so answers can be traced.
