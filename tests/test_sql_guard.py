import pytest

from geomlab.tools.sql_guard import SQLRejected, validate


def test_allowed_query_gets_limit():
    sql = validate("select district, avg(final_score) from analytics.scores where business_type='cafe' group by 1")
    assert "LIMIT 200" in sql


def test_cte_on_allowed_views():
    validate("with t as (select * from analytics.businesses) select count(*) from t")


@pytest.mark.parametrize("sql", [
    "select * from serving.scores",
    "select * from pg_catalog.pg_roles",
    "delete from analytics.scores",
    "select 1; drop table analytics.scores",
    "select pg_sleep(10) from analytics.scores",
    "insert into analytics.scores select * from analytics.scores",
    "select * from analytics.scores s join gold.cell_features f using (h3_id)",
])
def test_rejected(sql):
    with pytest.raises(SQLRejected):
        validate(sql)


def test_large_limit_is_capped():
    assert "LIMIT 200" in validate("select * from analytics.districts limit 100000")
