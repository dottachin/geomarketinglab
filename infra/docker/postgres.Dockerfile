# PostGIS + pgvector in one instance (doc §7.2: one database to operate, hybrid spatial + semantic filters)
FROM postgis/postgis:17-3.6
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-17-pgvector \
 && rm -rf /var/lib/apt/lists/*
