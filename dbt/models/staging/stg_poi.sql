select * exclude (geometry) from {{ source('silver', 'poi') }}
