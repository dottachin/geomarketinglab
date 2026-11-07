-- gold.training_set: one row per (cell, business type) among candidate cells.
-- label = 1 when the hexagon already contains at least one business of that type
-- (doc §6.1 "presence prediction"; survivor-bias caveat documented in docs/methodology.md).
select
    f.city, f.h3_id, c.business_type,
    (c.n_in_cell > 0)::integer as label,
    c.n_in_cell, c.comp_500m, c.comp_1000m, c.comp_catchment, c.nearest_comp_m, c.pop_catchment,
    f.data_date
from {{ ref('gold_cell_features') }} f
join {{ ref('stg_cell_competition') }} c on c.city = f.city and c.h3_id = f.h3_id
where f.is_candidate
