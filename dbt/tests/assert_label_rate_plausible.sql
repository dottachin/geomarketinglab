-- Guard against broken labels: per business type, positives must be between 0.1 % and 60 %.
{{ config(severity='warn') }}
select business_type, avg(label) as rate
from {{ ref('gold_training_set') }}
group by business_type
having avg(label) < 0.001 or avg(label) > 0.6
