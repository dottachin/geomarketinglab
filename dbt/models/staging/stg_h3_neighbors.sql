select * from {{ source('silver', 'h3_neighbors') }}
