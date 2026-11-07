-- GeoMarketingLab serving database (doc §2, §7.2, §8): PostGIS + pgvector in one instance.
-- Applied idempotently by `geomlab db-init` (owner role). {embedding_dim} is substituted.
--
-- schemas
--   serving    scores, SHAP drivers, hexagons, POIs, active model   (read by the tools)
--   analytics  curated read-only views — the ONLY objects query_stats may touch
--   rag        documents + chunk embeddings for search_documents
--   gold       wide copy of gold.cell_features (raw feature values for explanations)

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS serving;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS serving.cells (
    h3_id      text PRIMARY KEY,
    city       text NOT NULL,
    district   text,
    area_m2    double precision,
    geom       geometry(Geometry, 4326) NOT NULL,
    centroid   geometry(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS cells_city_idx ON serving.cells (city);
CREATE INDEX IF NOT EXISTS cells_geom_idx ON serving.cells USING gist (geom);

CREATE TABLE IF NOT EXISTS serving.pois (
    city      text NOT NULL,
    osm_id    text NOT NULL,
    role      text NOT NULL,
    category  text NOT NULL,
    name      text,
    h3_id     text,
    geom      geometry(Point, 4326) NOT NULL,
    PRIMARY KEY (city, osm_id)
);
CREATE INDEX IF NOT EXISTS pois_geom_idx ON serving.pois USING gist (geom);
CREATE INDEX IF NOT EXISTS pois_cat_idx ON serving.pois (city, category);

CREATE TABLE IF NOT EXISTS serving.scores (
    h3_id                 text NOT NULL,
    city                  text NOT NULL,
    business_type         text NOT NULL,
    model_version         text NOT NULL,
    data_date             date NOT NULL,
    suitability           real NOT NULL,     -- calibrated probability
    raw_score             real NOT NULL,
    saturation            real NOT NULL,     -- 0..1
    final_score           real NOT NULL,     -- suitability × (1 − λ·saturation)
    rank                  integer NOT NULL,  -- within (city, business_type, model_version)
    percentile            real NOT NULL,
    n_existing            integer NOT NULL,
    competitors_catchment integer NOT NULL,
    nearest_competitor_m  real,
    pop_catchment         real,
    opportunity           boolean NOT NULL,  -- high suitability and no existing business here
    low_confidence        boolean NOT NULL,
    confidence_reasons    text[] NOT NULL DEFAULT '{}',
    scored_at             timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (h3_id, business_type, model_version)
);
CREATE INDEX IF NOT EXISTS scores_lookup_idx ON serving.scores (model_version, city, business_type, rank);

CREATE TABLE IF NOT EXISTS serving.shap_drivers (
    h3_id          text NOT NULL,
    business_type  text NOT NULL,
    model_version  text NOT NULL,
    rank           smallint NOT NULL,
    feature        text NOT NULL,
    value          double precision,
    shap           real NOT NULL,
    PRIMARY KEY (h3_id, business_type, model_version, rank)
);

CREATE TABLE IF NOT EXISTS serving.active_model (
    id            smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model_version text NOT NULL,
    family        text NOT NULL,
    data_date     date NOT NULL,
    metrics       jsonb NOT NULL DEFAULT '{}',
    activated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag.documents (
    doc_id      text PRIMARY KEY,
    title       text NOT NULL,
    source_path text NOT NULL,
    sha256      text NOT NULL,
    doc_type    text,
    city        text,
    year        integer,
    lang        text,
    n_pages     integer,
    indexed_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag.chunks (
    chunk_id   bigserial PRIMARY KEY,
    doc_id     text NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    page       integer,
    section    text,
    content    text NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(section, '') || ' ' || content)) STORED,
    embedding  vector({embedding_dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON rag.chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON rag.chunks USING hnsw (embedding vector_cosine_ops);

-- analytics views (query_stats allow-list). Recreated after every scoring run.
CREATE OR REPLACE VIEW analytics.scores AS
SELECT s.city, s.h3_id, c.district, s.business_type, s.final_score, s.suitability, s.saturation,
       s.rank, s.n_existing, s.competitors_catchment, s.pop_catchment, s.opportunity,
       s.low_confidence, s.model_version, s.data_date
FROM serving.scores s
JOIN serving.active_model a ON a.model_version = s.model_version
JOIN serving.cells c ON c.h3_id = s.h3_id;

CREATE OR REPLACE VIEW analytics.businesses AS
SELECT p.city, c.district, p.category AS business_type, p.name, p.h3_id
FROM serving.pois p LEFT JOIN serving.cells c ON c.h3_id = p.h3_id
WHERE p.role = 'business';

CREATE OR REPLACE VIEW analytics.districts AS
SELECT c.city, c.district, count(*) AS n_cells, sum(c.area_m2) / 1e6 AS area_km2
FROM serving.cells c GROUP BY c.city, c.district;
