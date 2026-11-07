{# Register RustFS (S3 API) credentials for delta_scan when the lake lives in object storage. #}
{% macro configure_lake() %}
  {% set lake = env_var('LAKE_URI', '') %}
  {% if lake.startswith('s3://') %}
    {% set endpoint = env_var('S3_ENDPOINT_URL', '') %}
    create or replace secret lake_s3 (
      type s3,
      key_id '{{ env_var("AWS_ACCESS_KEY_ID", "") }}',
      secret '{{ env_var("AWS_SECRET_ACCESS_KEY", "") }}',
      region '{{ env_var("AWS_REGION", "us-east-1") }}'
      {% if endpoint %}
      , endpoint '{{ endpoint.split("://")[-1] }}'
      , url_style 'path'
      , use_ssl {{ 'false' if endpoint.startswith('http://') else 'true' }}
      {% endif %}
    )
  {% else %}
    select 1
  {% endif %}
{% endmacro %}

{# Use the custom schema name as-is (silver / gold) instead of <target>_<schema>. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
  {{ custom_schema_name if custom_schema_name else target.schema }}
{%- endmacro %}
