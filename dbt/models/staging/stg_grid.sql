select * exclude (geometry) from {{ source('silver', 'grid') }}
