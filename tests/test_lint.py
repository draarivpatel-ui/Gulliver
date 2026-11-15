import pytest

from gulliver.lint import lint
from gulliver.sqlsafety import classify


def rules(sql, dialect="duckdb", table_info=None):
    stmt = classify(sql, dialect)
    return {f.rule: f.severity for f in lint(stmt.select, dialect, table_info)}


def test_clean_aggregate_has_no_findings():
    assert rules("select country, count(*) from users where signup_date >= '2024-01-01' group by 1") == {}


def test_unbounded_output_absent_when_the_outer_query_aggregates():
    for sql in ("select count(*) from events",
                "select country, count(*) from users group by 1",
                "select exists (select 1 from orders) as any_orders",
                "select (select count(*) from events) as n",
                "select 1"):
        for dialect in ("duckdb", "postgres"):
            assert "GV002" not in rules(sql, dialect), (sql, dialect)


def test_aggregate_in_a_scalar_subquery_does_not_collapse_the_outer_rows():
    # The COUNT(*) belongs to the subquery: the outer query still returns one row per user.
    for dialect in ("duckdb", "postgres"):
        r = rules("select u.id, (select count(*) from orders o where o.user_id = u.id) as n from users u",
                  dialect)
        assert r["GV002"] == "warn", (r, dialect)


def test_unbounded_output_still_reported_for_a_plain_select():
    for dialect in ("duckdb", "postgres"):
        assert rules("select id from users", dialect)["GV002"] == "warn"


def test_select_star_and_unbounded_output():
    r = rules("select * from events")
    assert r["GV001"] == "warn"
    assert r["GV002"] == "warn"


def test_exists_subquery_star_is_not_a_scan():
    for dialect in ("duckdb", "postgres"):
        sql = "select u.id from users u where exists (select * from orders o where o.user_id = u.id) limit 10"
        assert "GV001" not in rules(sql, dialect), dialect
        not_exists = sql.replace("exists", "not exists")
        assert "GV001" not in rules(not_exists, dialect), dialect


def test_star_after_an_exists_subquery_is_still_found():
    derived = ("select u.id from users u where exists (select * from orders o where o.user_id = u.id) "
               "and u.id in (select * from ids) limit 10")
    assert rules(derived)["GV001"] == "warn"
    nested = ("select u.id from users u where exists (select * from orders o where o.user_id = u.id) "
              "and exists (select 1 from flags f where f.user_id = u.id) limit 10")
    assert "GV001" not in rules(nested)


def test_plain_and_derived_table_star_still_warn():
    assert rules("select * from users limit 10")["GV001"] == "warn"
    assert rules("select id from (select * from users) t limit 10")["GV001"] == "warn"
    assert rules("select * from users limit 10", "postgres")["GV001"] == "warn"


def test_limit_removes_unbounded_warning():
    assert "GV002" not in rules("select id from events limit 10")


def test_missing_join_condition_is_an_error():
    assert rules("select e.id from events e, users u")["GV003"] == "error"


def test_natural_join_has_an_implicit_key_and_is_not_cartesian():
    for sql in ("select a from t natural join u limit 5",
                "select a from t natural left join u limit 5"):
        for dialect in ("duckdb", "postgres"):
            assert "GV003" not in rules(sql, dialect), (sql, dialect)


def test_comma_join_with_where_key_is_info():
    assert rules("select e.id from events e, users u where e.user_id = u.user_id limit 5")["GV003"] == "info"


def test_cross_join_is_a_warning():
    assert rules("select e.id from events e cross join users u limit 5")["GV003"] == "warn"


def test_unnest_join_is_not_cartesian():
    r = rules("select x from t, unnest(t.arr) as x limit 5", "bigquery")
    assert "GV003" not in r


def test_non_equi_join():
    assert rules("select a.id from a join b on a.ts between b.start_ts and b.end_ts limit 5")["GV008"] == "warn"


@pytest.mark.parametrize("on", ["1=1", "true", "'a' = 'a'", "t.id = t.id", "u.x = u.y", "t.id = t.id and 1=1"])
def test_join_condition_that_does_not_link_the_tables_is_a_cartesian_error(on):
    r = rules(f"select t.a from t join u on {on} limit 5")
    assert r["GV003"] == "error"
    assert "GV008" not in r


@pytest.mark.parametrize("on", [
    "t.id = u.id",
    "t.id = u.id and 1=1",
    "t.id = u.id and u.kind = 'x'",
    "1=1 and t.id = u.id",
    "id = user_id",
])
def test_join_condition_with_a_real_key_or_unknown_columns_is_not_flagged(on):
    r = rules(f"select t.a from t join u on {on} limit 5")
    assert "GV003" not in r


def test_join_using_and_cross_join_behave_as_before():
    assert "GV003" not in rules("select a from t join u using (id) limit 5")
    assert rules("select a from t cross join u limit 5")["GV003"] == "warn"


def test_unlinked_condition_message_names_the_condition():
    stmt = classify("select t.a from t join u on t.id = t.id limit 5", "duckdb")
    finding, = [f for f in lint(stmt.select, "duckdb") if f.rule == "GV003"]
    assert "does not link the two tables" in finding.message
    assert "t.id = t.id" in finding.message


def test_missing_partition_filter_uses_metadata():
    info = {"events": {"partitioned_by": "event_date"}}
    assert rules("select count(*) from events", table_info=info)["GV004"] == "error"
    assert "GV004" not in rules("select count(*) from events where event_date = '2026-01-01'", table_info=info)


def test_function_on_filter_column():
    assert rules("select count(*) from events where date(ts) = '2024-01-01'")["GV005"] == "warn"
    assert "GV005" not in rules("select count(*) from events where ts >= '2024-01-01'")


def gv005(sql, dialect="duckdb"):
    stmt = classify(sql, dialect)
    found = [f for f in lint(stmt.select, dialect) if f.rule == "GV005"]
    assert len(found) == 1, (sql, dialect, found)
    return found[0]


GV005_MESSAGE = "The filter wraps {} in a function or calculation, which blocks partition pruning and min/max skipping."
GV005_DIALECTS = ("duckdb", "postgres", "bigquery", "snowflake", "sqlite")
GV005_FORMAT_DIALECTS = (*GV005_DIALECTS, "mysql")


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
@pytest.mark.parametrize("where, column", [
    ("date(created_at) = '2024-01-01'", "created_at"),
    ("cast(created_at as date) = '2024-01-01'", "created_at"),
    ("{trunc} = '2024-01-01'", "created_at"),
    ("extract(year from created_at) = 2024", "created_at"),
])
def test_gv005_fix_for_date_functions_names_the_column(dialect, where, column):
    trunc = "date_trunc(created_at, day)" if dialect == "bigquery" else "date_trunc('day', created_at)"
    f = gv005(f"select a from t where {where.format(trunc=trunc)} limit 5", dialect)
    assert (f.severity, f.message) == ("warn", GV005_MESSAGE.format(column))
    assert f.fix == (f"Compare the raw column to a constant range, e.g. {column} >= '2024-01-01' AND "
                     f"{column} < '2024-01-02' instead of DATE({column}) = '2024-01-01'.")


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
@pytest.mark.parametrize("where", [
    "amount * 1 > 9007199254740992",  # safe integer-looking factor can still be unsafe by column type/precision
    "amount * 2 > 1 + 2",             # compound constant must not lose parentheses in any concrete rewrite
    "amount + 5 = '20'",              # nonnumeric comparison constant must not get broken arithmetic SQL
    "amount * 1.2 > 100",
    "100 < amount * 1.2",
    "amount / 4 <= 10",
    "amount * -2 > 10",
    "amount * 0 > 10",
    "3 - amount > 1",
    "amount * 2 > other_amount",
    "amount * 2 between 1 and 5",
])
def test_gv005_fix_for_arithmetic_is_safe_general_advice(dialect, where):
    f = gv005(f"select a from t where {where} limit 5", dialect)
    assert f.message == GV005_MESSAGE.format("amount")
    assert "Move the constant arithmetic to the other side of the comparison when types and overflow allow" in f.fix
    assert "raw column amount" in f.fix
    assert "Do not emit a concrete rewritten comparison" in f.fix
    assert "amount > 9007199254740992 / 1.0" not in f.fix
    assert "1 + 2 / 2.0" not in f.fix
    assert "amount = '20' - 5" not in f.fix


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
@pytest.mark.parametrize("where", [
    "coalesce(n, 1) = 1.00000000000000001",
    "coalesce(n, 0e0) = 9007199254740993",
    "coalesce(n, '0') = '0'",
    "coalesce(status, 'x') = 'x'",
    "coalesce(status, 'x') = 'y'",
])
def test_gv005_fix_for_coalesce_is_safe_general_advice(dialect, where):
    column = "status" if "status" in where else "n"
    f = gv005(f"select a from t where {where} limit 5", dialect)
    assert f.message == GV005_MESSAGE.format(column)
    assert f"Compare the raw column {column} directly and handle NULL explicitly" in f.fix
    assert "Do not drop COALESCE" in f.fix
    assert "IS NULL" not in f.fix and "IS NOT NULL" not in f.fix
    assert "(n = " not in f.fix and "(status = " not in f.fix


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
def test_gv005_fix_for_case_functions_preserves_exact_transformation(dialect):
    f_upper = gv005("select a from t where upper(name) like 'A%' limit 5", dialect)
    assert f_upper.message == GV005_MESSAGE.format("name")
    assert "ILIKE" not in f_upper.fix
    assert "UPPER(name) runs on every row and forces a full scan." in f_upper.fix
    assert "precomputed upper-case copy of name (UPPER(name))" in f_upper.fix
    assert "lower-case" not in f_upper.fix

    f_lower = gv005("select a from t where lower(name) = 'a' limit 5", dialect)
    assert f_lower.message == GV005_MESSAGE.format("name")
    assert "ILIKE" not in f_lower.fix
    assert "LOWER(name) runs on every row and forces a full scan." in f_lower.fix
    assert "precomputed lower-case copy of name (LOWER(name))" in f_lower.fix
    assert "upper-case" not in f_lower.fix

    f_pat = gv005("select a from t where 'ALICE' like upper(pattern) limit 5", dialect)
    assert "ILIKE" not in f_pat.fix
    assert "precomputed upper-case copy of pattern (UPPER(pattern))" in f_pat.fix
    assert "operand order" in f_pat.fix


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
def test_gv005_fix_for_trim(dialect):
    f = gv005("select a from t where trim(name) = 'x' limit 5", dialect)
    assert f.fix.startswith("TRIM(name) runs on every row and forces a full scan.") and "ILIKE" not in f.fix
    assert "type affinity" in f.fix
    f_opts = gv005("select a from t where trim(code, '0') = '1' limit 5", dialect)
    assert "ILIKE" not in f_opts.fix
    assert "TRIM(code) runs on every row" not in f_opts.fix
    assert "'0'" in f_opts.fix


@pytest.mark.parametrize("dialect", GV005_FORMAT_DIALECTS)
@pytest.mark.parametrize("where, column", [
    ("to_char(amount, 'FM999MI') = '123'", "amount"),
    ("to_char(amount, 'FM999%MI') = '123'", "amount"),
    ("to_char(amount, '999%%') = '123'", "amount"),
    ("to_char(amt, '999D99') = '123.45'", "amt"),
    ("to_char(amount, '999TH') = '123RD'", "amount"),
    ("to_char(amount, '990') = '123'", "amount"),
    ("to_char(amount, '999.00') = '123.45'", "amount"),
    ("to_char(amount, '%M99') = '123'", "amount"),
    ("to_char(amount, '%u99') = '123'", "amount"),
    ("to_char(amount, '%OM99') = '123'", "amount"),
    ("to_char(amount, '%Ou99') = '123'", "amount"),
    ("to_char(amount, 'FM999\"YYYY\"') = '123'", "amount"),
    ("to_char(amount, 'FM999\"YYYYMMDD\"') = '123'", "amount"),
    ("to_char(amount, 'MI') = '-'", "amount"),
    ("to_char(amount, 'D') = '1'", "amount"),
    ("to_char(amount, '%M') = '1'", "amount"),
    ("to_char(amount, '%u') = '1'", "amount"),
    ("to_char(amount, '%OM') = '1'", "amount"),
    ("to_char(amount, '%Ou') = '1'", "amount"),
    ("to_char(x, 'JSON') = '1'", "x"),
    ('to_char(amount, \'"label \\\"MM\\\""999\') = \'1\'', "amount"),
])
def test_gv005_numeric_quoted_and_unknown_tochar_formats_do_not_get_date_advice(dialect, where, column):
    f = gv005(f"select a from t where {where} limit 5", dialect)
    assert f.message == GV005_MESSAGE.format(column)
    assert "constant range" not in f.fix
    assert "DATE(" not in f.fix
    assert "2024-01-01" not in f.fix
    assert f.fix == (f"Compare the raw column {column} where possible, or filter on a partition or indexed column first "
                     "so less data is read.")


@pytest.mark.parametrize("dialect", GV005_FORMAT_DIALECTS)
@pytest.mark.parametrize("fmt", ["%a", "%B", "%Y-%m-%d", "%Y-%m-%d 00:00", "%-d", "%02m-%d 09:00", "%M99", "%u99"])
def test_gv005_timetostr_temporal_directives_keep_date_advice(dialect, fmt):
    f = gv005(f"select a from t where strftime(ts, '{fmt}') = 'x' limit 5", dialect)
    assert f.message == GV005_MESSAGE.format("ts")
    assert f.fix == ("Compare the raw column to a constant range, e.g. ts >= '2024-01-01' AND "
                     "ts < '2024-01-02' instead of DATE(ts) = '2024-01-01'.")


@pytest.mark.parametrize("dialect", GV005_FORMAT_DIALECTS)
@pytest.mark.parametrize("fmt", ["%Y-%m-%d 00:00", "%-d"])
def test_gv005_format_timestamp_with_literals_and_flags_keeps_date_advice(dialect, fmt):
    f = gv005(f"select a from t where format_timestamp('{fmt}', ts) = 'x' limit 5", dialect)
    assert f.message == GV005_MESSAGE.format("ts")
    assert f.fix == ("Compare the raw column to a constant range, e.g. ts >= '2024-01-01' AND "
                     "ts < '2024-01-02' instead of DATE(ts) = '2024-01-01'.")


@pytest.mark.parametrize("dialect", GV005_FORMAT_DIALECTS)
@pytest.mark.parametrize("fmt", [
    "YYYY-MM-DD", "YYYY-MM-DD 00:00", "DDTH", "DDTH Month YYYY", "MONTH", "YYYYMMDD", "HH24MISS",
    "YYYYMMDDHH24MISS", "YYYYMMDDHH24MISSFF0", "HH24MISSFF0", "YYYYMMDDHH24MISSFF6",
    "YYYY_MM_DD", "HH24_MI_SS", "FMYYYYMMDD", "FMDDTHMonthYYYY", "yyyymmdd", "hh24miss",
])
def test_gv005_to_char_temporal_format_keeps_date_advice(dialect, fmt):
    f = gv005(f"select a from t where to_char(created_at, '{fmt}') = 'x' limit 5", dialect)
    assert f.fix == ("Compare the raw column to a constant range, e.g. created_at >= '2024-01-01' AND "
                     "created_at < '2024-01-02' instead of DATE(created_at) = '2024-01-01'.")


@pytest.mark.parametrize("dialect", GV005_FORMAT_DIALECTS)
@pytest.mark.parametrize("expr", [
    "to_char(cast(created_at as timestamp), 'MI')",
    "to_char(date(created_at), 'MI')",
    "to_char(cast(created_at as date), 'D')",
    "to_char(cast(created_at as timestamp), '%M')",
])
def test_gv005_to_char_ambiguous_format_with_temporal_operand_keeps_date_advice(dialect, expr):
    f = gv005(f"select a from t where {expr} = '1' limit 5", dialect)
    assert f.fix == ("Compare the raw column to a constant range, e.g. created_at >= '2024-01-01' AND "
                     "created_at < '2024-01-02' instead of DATE(created_at) = '2024-01-01'.")


@pytest.mark.parametrize("dialect", GV005_DIALECTS)
@pytest.mark.parametrize("where", [
    "substr(code, 1, 2) = 'NL'", "cast(code as int) = 3", "length(code) > 3",
])
def test_gv005_fix_for_other_functions_is_neutral(dialect, where):
    f = gv005(f"select a from t where {where} limit 5", dialect)
    assert f.message == GV005_MESSAGE.format("code")
    assert f.fix == ("Compare the raw column code where possible, or filter on a partition or indexed column first "
                     "so less data is read.")


@pytest.mark.parametrize("where", ["upper(name) like 'A%'", "amount * 1.2 > 100", "coalesce(status, 'x') = 'x'",
                                   "substr(code, 1, 2) = 'NL'", "trim(name) = 'x'", "amount * -2 > 10"])
def test_gv005_fix_mentions_no_date_unless_the_filter_is_about_time(where):
    for dialect in GV005_DIALECTS:
        fix = gv005(f"select a from t where {where} limit 5", dialect).fix
        assert "ts" not in fix.split() and "2024" not in fix and "DATE" not in fix, (where, dialect, fix)


def test_order_without_limit():
    assert rules("select id from events order by id")["GV006"] == "warn"


@pytest.mark.parametrize("dialect, function", [("duckdb", "random()"), ("postgres", "random()"),
                                               ("bigquery", "rand()")])
@pytest.mark.parametrize("suffix", ["", " limit 100"])
def test_order_by_random_warns_even_with_limit(dialect, function, suffix):
    stmt = classify(f"select id from events order by {function}{suffix}", dialect)
    finding, = [f for f in lint(stmt.select, dialect) if f.rule == "GV016"]
    assert finding.severity == "warn"
    assert finding.name == "order-by-random"
    assert "TABLESAMPLE" in finding.fix and "USING SAMPLE" in finding.fix
    assert finding.node is not None


@pytest.mark.parametrize("sql", [
    "select random() as r from events limit 10",
    "select id from events where random() < 0.1 limit 10",
    "select random() as r from events order by id limit 10",
    "select id from events order by id limit 10",
])
def test_random_outside_order_by_does_not_warn(sql):
    assert "GV016" not in rules(sql)


def test_order_by_random_inside_subquery_and_expression():
    sql = "select id from (select id from events order by abs(random()) limit 100) sample limit 10"
    assert rules(sql)["GV016"] == "warn"


def test_order_by_random_is_listed_with_rule_metadata():
    from gulliver.lint_report import rules as rule_docs

    rule, = [r for r in rule_docs() if r["rule"] == "GV016"]
    assert rule["name"] == "order-by-random"
    assert rule["default_severity"] == "warn"
    assert rule["autofix"] == "none"
    assert "TABLESAMPLE" in rule["fix"] and "USING SAMPLE" in rule["fix"]


def test_count_distinct_union_window_not_in():
    r = rules("select count(distinct user_id) from events")
    assert r["GV007"] == "info"
    assert rules("select id from a union select id from b")["GV009"] == "info"
    assert rules("select id, row_number() over (order by ts) from events limit 5")["GV011"] == "warn"
    assert rules("select id from a where id not in (select id from b) limit 5")["GV012"] == "info"


def test_scalar_subquery_in_select():
    assert rules("select id, (select max(x) from b where b.id = a.id) from a limit 5")["GV010"] == "info"


def test_leading_wildcard_like_is_info():
    assert rules("select a from t where b like '%x'")["GV013"] == "info"
    assert rules("select a from t where b like '_x'")["GV013"] == "info"
    assert rules("select a from t where b ilike '%x%' limit 5")["GV013"] == "info"
    assert rules("select a from t where b ilike '__x%' limit 5")["GV013"] == "info"
    assert "GV013" not in rules("select a from t where b like 'x%'")
    assert "GV013" not in rules("select a from t where b like 'a_c'")
    # an escaped underscore is a literal prefix, not a wildcard
    assert "GV013" not in rules("select a from t where b like '\\_x' escape '\\'")


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "bigquery"])
def test_deep_offset_flagged_across_dialects(dialect):
    stmt = classify("select id from events order by id limit 50 offset 200000", dialect)
    findings = [f for f in lint(stmt.select, dialect) if f.rule == "GV017"]
    assert len(findings) == 1
    assert findings[0].severity == "warn"
    assert findings[0].name == "deep-offset"
    assert "keyset pagination" in findings[0].fix.lower()
    assert findings[0].node is not None


def test_deep_offset_fetch_next_postgres():
    stmt = classify("select id from events offset 200000 rows fetch next 50 rows only", "postgres")
    findings = [f for f in lint(stmt.select, "postgres") if f.rule == "GV017"]
    assert len(findings) == 1
    assert findings[0].severity == "warn"
    assert "keyset pagination" in findings[0].fix.lower()


def test_deep_offset_in_subquery():
    stmt = classify("select * from (select id from events offset 200000)", "duckdb")
    findings = [f for f in lint(stmt.select, "duckdb") if f.rule == "GV017"]
    assert len(findings) == 1
    assert findings[0].severity == "warn"


@pytest.mark.parametrize("sql", [
    "select id from events offset 100",
    "select id from events offset 9999",
    "select id from events offset ?",
    "select id from events limit 50",
])
def test_deep_offset_negatives(sql):
    stmt = classify(sql, "duckdb")
    findings = [f for f in lint(stmt.select, "duckdb") if f.rule == "GV017"]
    assert len(findings) == 0


COUNT_SUBQUERY = "(select count(*) from orders o where o.user_id = u.id)"


@pytest.mark.parametrize("dialect", ["duckdb", "postgres"])
@pytest.mark.parametrize("sql", [
    f"select u.id from users u where {COUNT_SUBQUERY} > 0 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} >= 1 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} <> 0 limit 10",
    f"select u.id from users u where 0 < {COUNT_SUBQUERY} limit 10",
    "select u.id from users u where (select count(1) from orders o where o.user_id = u.id) > 0 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} = 0 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} < 1 limit 10",
])
def test_count_for_existence_flagged(sql, dialect):
    stmt = classify(sql, dialect)
    findings = [f for f in lint(stmt.select, dialect) if f.rule == "GV018"]
    assert len(findings) == 1, (sql, dialect, findings)
    assert findings[0].severity == "info"
    assert findings[0].name == "count-for-existence"
    assert findings[0].node is not None


def test_count_for_existence_suggests_exists():
    stmt = classify(f"select u.id from users u where {COUNT_SUBQUERY} > 0 limit 10", "duckdb")
    finding, = [f for f in lint(stmt.select, "duckdb") if f.rule == "GV018"]
    assert "EXISTS" in finding.fix
    assert "NOT EXISTS" not in finding.fix


def test_count_for_existence_suggests_not_exists():
    stmt = classify(f"select u.id from users u where {COUNT_SUBQUERY} = 0 limit 10", "duckdb")
    finding, = [f for f in lint(stmt.select, "duckdb") if f.rule == "GV018"]
    assert "NOT EXISTS" in finding.fix


def test_count_for_existence_in_case_when():
    # GV010 may fire on the same subquery; that is fine.
    sql = "select case when (select count(*) from orders o) > 0 then 1 else 0 end as n from users limit 10"
    assert rules(sql)["GV018"] == "info"


@pytest.mark.parametrize("sql", [
    f"select u.id from users u where {COUNT_SUBQUERY} > 5 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} = 3 limit 10",
    f"select u.id from users u where {COUNT_SUBQUERY} >= 2 limit 10",
    "select u.id from users u where (select count(*) from orders o group by o.user_id) > 0 limit 10",
    "select u.id from users u where (select count(o.id) from orders o) > 0 limit 10",
    "select user_id from orders group by user_id having count(*) > 0",
    "select count(*) from users",
    "select u.id from users u where exists (select 1 from orders o where o.user_id = u.id) limit 10",
])
def test_count_for_existence_not_flagged(sql):
    assert "GV018" not in rules(sql), sql


def test_count_for_existence_is_listed_with_rule_metadata():
    from gulliver.lint_report import rules as rule_docs

    rule, = [r for r in rule_docs() if r["rule"] == "GV018"]
    assert rule["name"] == "count-for-existence"
    assert rule["default_severity"] == "info"
    assert rule["autofix"] == "none"
    assert "EXISTS" in rule["fix"]


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "bigquery", "snowflake"])
@pytest.mark.parametrize("comparison", [
    "amount = NULL", "amount <> NULL", "amount != NULL",
    "NULL = amount", "NULL <> amount", "NULL != amount", "NULL = NULL",
    "amount = (NULL)",
])
def test_null_comparison_warns(comparison, dialect):
    stmt = classify(f"select event_id from events where {comparison} limit 5", dialect)
    finding, = [f for f in lint(stmt.select, dialect) if f.rule == "GV020"]
    assert finding.name == "null-comparison"
    assert finding.severity == "warn"
    assert finding.message == (
        f"{comparison.replace('!=', '<>')} is never true: a comparison with NULL gives NULL."
    )
    assert "IS NULL" in finding.fix and "IS NOT NULL" in finding.fix
    assert "IS NOT DISTINCT FROM" in finding.fix
    assert finding.node is not None


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "bigquery", "snowflake"])
@pytest.mark.parametrize("comparison", [
    "u.deleted_at = NULL", "NULL = u.deleted_at", "u.deleted_at <> NULL",
])
def test_null_comparison_message_preserves_join_operands(comparison, dialect):
    stmt = classify(
        f"select e.event_id from events e join users u on e.user_id = u.id and {comparison} limit 5",
        dialect,
    )
    finding, = [f for f in lint(stmt.select, dialect) if f.rule == "GV020"]
    assert finding.message == f"{comparison} is never true: a comparison with NULL gives NULL."


@pytest.mark.parametrize("sql", [
    "select e.event_id from events e join users u on e.user_id = u.id and e.amount = NULL limit 5",
    "select case when NULL != amount then 1 else 0 end as flag from events limit 5",
    "select event_id from events where exists (select 1 from users where id <> NULL) limit 5",
])
def test_null_comparison_in_join_case_and_subquery(sql):
    assert rules(sql)["GV020"] == "warn"


@pytest.mark.parametrize("comparison", [
    "amount IS NULL", "amount IS NOT NULL", "amount IS NOT DISTINCT FROM NULL",
    "amount IS DISTINCT FROM NULL", "amount IN (1, NULL)",
    "amount = 1", "amount <> 1", "amount = 'NULL'", "amount = coalesce(NULL, 1)",
])
def test_null_comparison_leaves_other_comparisons_alone(comparison):
    assert "GV020" not in rules(f"select event_id from events where {comparison} limit 5")


def test_null_comparison_is_listed_without_autofix():
    from gulliver.fixer import rewrite
    from gulliver.lint_report import rules as rule_docs

    rule, = [r for r in rule_docs() if r["rule"] == "GV020"]
    assert rule["name"] == "null-comparison"
    assert rule["default_severity"] == "warn"
    assert rule["autofix"] == "none"
    sql = "select event_id from events where amount = NULL limit 5"
    for mode in ("safe", "all"):
        result = rewrite(sql, mode=mode)
        assert result["changes"] == []
