"""Validation for query_stats (doc §7.1 "text-to-SQL with read-only role and schema context").

Defence in depth:
  1. parse with sqlglot (PostgreSQL dialect): exactly one SELECT statement, no DML/DDL;
  2. every referenced table must be in the analytics.* allow-list;
  3. a LIMIT is enforced;
  4. dangerous functions are rejected;
  5. execution uses the ``geomlab_sql`` role (SELECT on analytics views only) with a
     5-second statement timeout — even a query that slipped through cannot write or read
     anything else.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "analytics.scores": "city, h3_id, district, business_type, final_score, suitability, saturation, rank, "
                        "n_existing, competitors_catchment, pop_catchment, opportunity, low_confidence, "
                        "model_version, data_date (one row per hexagon × business type, active model only)",
    "analytics.businesses": "city, district, business_type, name, h3_id (existing businesses mapped in OSM)",
    "analytics.districts": "city, district, n_cells, area_km2",
}
FORBIDDEN_FUNCS = {"pg_sleep", "dblink", "lo_import", "lo_export", "pg_read_file", "pg_ls_dir",
                   "set_config", "current_setting", "pg_terminate_backend", "copy"}
MAX_ROWS = 200


class SQLRejected(ValueError):
    pass


def validate(sql: str) -> str:
    """Return a safe, LIMIT-bounded SQL string or raise SQLRejected."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLRejected(f"could not parse SQL: {exc}") from None
    if len(statements) != 1:
        raise SQLRejected("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise SQLRejected("only SELECT queries are allowed")
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                             exp.Command, exp.Into)):
            raise SQLRejected("only read-only SELECT queries are allowed")
        if isinstance(node, exp.Func):
            name = (node.sql_name() if not isinstance(node, exp.Anonymous) else node.name).lower()
            if name in FORBIDDEN_FUNCS:
                raise SQLRejected(f"function {name} is not allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not table.db and name in ctes:
            continue
        full = f"{table.db.lower()}.{name}" if table.db else name
        if full not in ALLOWED_TABLES:
            raise SQLRejected(f"table {full!r} is not allowed; use one of {sorted(ALLOWED_TABLES)}")
    limit = tree.args.get("limit")
    if limit is None or not isinstance(limit.expression, exp.Literal) or int(limit.expression.this) > MAX_ROWS:
        tree = tree.limit(MAX_ROWS)
    return tree.sql(dialect="postgres")


def schema_context() -> str:
    return "\n".join(f"- {t}: {cols}" for t, cols in ALLOWED_TABLES.items())
