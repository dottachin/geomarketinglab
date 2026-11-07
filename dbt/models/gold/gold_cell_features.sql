-- gold.cell_features: one documented, model-ready row per H3 cell (doc §4.2 gold layer).
-- Competition columns are deliberately absent (they only feed the saturation penalty).
with base as (select * from {{ ref('int_cell_base') }}),
ctx as (select * from {{ ref('int_cell_context') }})
select
    base.* exclude (hcp_pct_age_15_19, hcp_pct_age_20_24, hcp_pct_age_25_29, hcp_pct_age_30_34, hcp_taux_chomage),
    ctx.* exclude (city, h3_id),
    (case when not base.has_gee then 1 else 0 end) + (case when not base.has_hcp then 1 else 0 end) as missing_sources,
    (
        coalesce(base.pop_cell, 0) >= {{ var('min_population') }}
        or coalesce(base.att_total_1000m, 0) >= {{ var('min_attractors_1km') }}
        or coalesce(base.builtup_ratio, 0) >= {{ var('min_builtup_ratio') }}
    ) and coalesce(base.water_ratio, 0) < 0.5                     as is_candidate,
    current_date                                                   as data_date
from base
left join ctx on ctx.city = base.city and ctx.h3_id = base.h3_id
