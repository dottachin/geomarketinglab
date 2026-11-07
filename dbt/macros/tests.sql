{% test unique_combination(model, columns) %}
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}

{% test dbt_utils_expression(model, column_name, expression) %}
select * from {{ model }} where not ({{ column_name }} {{ expression }})
{% endtest %}
