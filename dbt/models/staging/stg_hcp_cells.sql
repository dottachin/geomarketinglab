select * from {{ source('silver', 'hcp_cells') }}
