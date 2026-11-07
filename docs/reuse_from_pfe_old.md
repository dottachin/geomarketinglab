# What was reused from `pfe-old/geomarketlab`

| Old | New | How |
|---|---|---|
| `src/utils/geo.py` (CRS discipline, H3 v4 helpers, metric buffers/counts/nearest) | `geomlab/geo.py` | ported; added weighted counts, neighbour pairs, point→cell |
| `src/utils/config.py` (YAML loaders, env settings) | `geomlab/config.py`, `geomlab/settings.py` | rewritten with pydantic-settings; business cases split into `business_types.yaml` + `poi_categories.yaml` taxonomy |
| `config/cities.yaml` | `config/cities.yaml` | ported (Marrakesh slug → `marrakech`, alias kept in tools) |
| `config/hcp.yaml` (file URLs, column positions, Casablanca aliases, geo-code prefixes) | `config/hcp.yaml` | copied as is + new `disaggregation` section |
| `src/data/build_grid.py` (clip, sliver removal, district assignment) | `geomlab/features/grid.py` | ported; boundaries now read from bronze |
| `src/data/extract_hcp.py` (parsers, geo-code normalisation, OSM name cleaning, alias matching, leaf preference) | `geomlab/ingestion/hcp.py` + `geomlab/features/hcp_silver.py` | parsers ported; centroid copy replaced by dasymetric disaggregation |
| `src/data/extract_gee.py` (S2 masking, Dynamic World, VIIRS, WorldPop, chunked reduceRegions, auth) | `geomlab/ingestion/gee.py` | ported per H3 cell; added VIIRS monthly stack for the trend, change token |
| `src/data/extract_osm.py` (OSMnx Overpass extraction) | `geomlab/ingestion/osm.py` + `features/osm_silver.py` | replaced by Geofabrik + diffs + pyosmium (doc §3.1); category derivation now from the taxonomy |
| `src/features/*` (competition rings, complementary POIs, road/intersection density, label builder, no-leakage rule) | `geomlab/features/families.py` + dbt gold | same ideas, families as in doc §5.2; leakage rule now enforced in code and in a dbt test |
| `src/models/train_ml.py` (spatial blocks, top-k metrics, SHAP) | `geomlab/ml/evaluation.py`, `ml/models/lgbm.py` | H3-parent GroupKFold with buffer instead of KMeans/district blocks; NDCG@k added |
| `src/bi/metrics.py` (saturation index) | `geomlab/ml/scoring.py::saturation` | redefined as competitors vs catchment capacity, bounded [0, 1) |
| `src/bi/explanations.py` (rule-based explanation text) | `tools/core.py::explain_location` + LLM | structured drivers; the LLM writes the text |
| `docker-compose.yml` profiles, PostGIS init | `infra/` | rebuilt: PostGIS + pgvector image, RustFS, Airflow 3, MLflow, Martin, web |
| `tests/conftest.py` synthetic fixtures | `tests/synthetic.py` | new synthetic city that runs the whole pipeline offline |

Not carried over (replaced by the new design): the ML/DL/Graph three-view fusion and agreement score,
CNN-on-patches + Grad-CAM (kept as a stretch goal), the Streamlit cockpit (replaced by React + chat),
the BI star schema views, the markdown report generator (replaced by the MCP `site_analysis_report`
prompt). The old processed data and caches were not copied (fresh extraction).
