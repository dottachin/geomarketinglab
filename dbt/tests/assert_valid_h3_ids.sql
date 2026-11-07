-- Every cell id must be a 15-hex-char H3 index.
select h3_id from {{ ref('gold_cell_features') }}
where not regexp_full_match(h3_id, '[0-9a-f]{15}')
