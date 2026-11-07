-- One row per (city, h3_id): every feature family joined, plus derived ratios.
with grid as (select * from {{ ref('stg_grid') }}),
gee as (select * exclude (feature_date) from {{ ref('stg_gee_cells') }}),
hcp as (select * from {{ ref('stg_hcp_cells') }})
select
    g.city, g.h3_id, g.district, g.lon, g.lat, g.area_m2,
    gee.* exclude (city, h3_id),
    gee.pop_cell / nullif(g.area_m2 / 1e6, 0)                       as pop_density_km2,
    d.* exclude (city, h3_id),
    e.* exclude (city, h3_id),
    a.* exclude (city, h3_id),
    acc.* exclude (city, h3_id),
    env.* exclude (city, h3_id),
    hcp.* exclude (city, h3_id),
    coalesce(hcp.hcp_pct_age_15_19, 0) + coalesce(hcp.hcp_pct_age_20_24, 0)
      + coalesce(hcp.hcp_pct_age_25_29, 0) + coalesce(hcp.hcp_pct_age_30_34, 0) as hcp_share_15_34,
    100 - hcp.hcp_taux_chomage                                      as hcp_employment_rate,
    gee.pop_cell is not null or gee.ndvi_mean is not null           as has_gee,
    coalesce(hcp.hcp_coverage, 0) > 0.5                             as has_hcp
from grid g
left join gee  on gee.city = g.city and gee.h3_id = g.h3_id
left join {{ ref('stg_cell_demand') }}        d   on d.city = g.city and d.h3_id = g.h3_id
left join {{ ref('stg_cell_economic') }}      e   on e.city = g.city and e.h3_id = g.h3_id
left join {{ ref('stg_cell_attractors') }}    a   on a.city = g.city and a.h3_id = g.h3_id
left join {{ ref('stg_cell_accessibility') }} acc on acc.city = g.city and acc.h3_id = g.h3_id
left join {{ ref('stg_cell_environment') }}   env on env.city = g.city and env.h3_id = g.h3_id
left join hcp on hcp.city = g.city and hcp.h3_id = g.h3_id
