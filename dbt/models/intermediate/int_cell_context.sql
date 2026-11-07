-- Spatial context (doc §5.2): mean of selected features over neighbours within k rings.
{%- set rings = var('context_rings') %}
{%- set feats = var('context_features') %}
with base as (select * from {{ ref('int_cell_base') }}),
nb as (select * from {{ ref('stg_h3_neighbors') }})
select
    nb.city, nb.h3_id
    {%- for k in rings %}{%- for f in feats %},
    avg(case when nb.distance <= {{ k }} then n.{{ f }} end) as ctx{{ k }}_{{ f }}
    {%- endfor %}{%- endfor %}
from nb
join base n on n.city = nb.city and n.h3_id = nb.neighbor_id
group by nb.city, nb.h3_id
