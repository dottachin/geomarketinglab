-- No-leakage rule: competition columns may never appear in the feature table.
select column_name
from information_schema.columns
where table_name = 'gold_cell_features'
  and (column_name like 'comp\_%' escape '\' or column_name = 'n_in_cell')
