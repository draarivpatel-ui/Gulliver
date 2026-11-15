import sqlite3

import duckdb
import pytest

from gulliver.lint import lint
from gulliver.sqlsafety import classify

DIALECTS = ("duckdb", "postgres", "bigquery", "snowflake", "sqlite")
FORMAT_DIALECTS = (*DIALECTS, "mysql")


def gv005_fix(sql, dialect="duckdb"):
    stmt = classify(sql, dialect)
    found = [f for f in lint(stmt.select, dialect) if f.rule == "GV005"]
    assert len(found) == 1, (sql, dialect, found)
    return found[0].fix


def rows(con, sql):
    return con.execute(sql).fetchall()


@pytest.mark.parametrize("dialect", DIALECTS)
def test_gv005_arithmetic_bigint_inversion_can_change_results_and_gets_safe_advice(dialect):
    # BIGINT 9007199254740993 * 1 > 9007199254740992 is true, while moving to the constant side
    # as 9007199254740992 / 1.0 introduces DOUBLE precision and is false in DuckDB.
    fix = gv005_fix("select id from t where amount * 1 > 9007199254740992", dialect)
    assert "Move the constant arithmetic to the other side of the comparison when types and overflow allow" in fix
    assert "raw column amount" in fix
    assert "9007199254740992 / 1.0" not in fix
    assert "amount > 9007199254740992 / 1.0" not in fix
    con = duckdb.connect()
    con.execute("create table t(id int, amount bigint)")
    con.execute("insert into t values (1, 9007199254740993)")
    original = "select id from t where amount * 1 > 9007199254740992 order by id"
    naive = "select id from t where amount > 9007199254740992 / 1.0 order by id"
    assert rows(con, original) == [(1,)]
    assert rows(con, naive) == []
    assert rows(con, original) != rows(con, naive)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_gv005_arithmetic_compound_constants_not_broken_by_missing_parentheses(dialect):
    # amount * 2 > 1 + 2 must not become amount > 1 + 2 / 2.0 (missing parens).
    fix = gv005_fix("select id from t where amount * 2 > 1 + 2", dialect)
    assert "Move the constant arithmetic to the other side of the comparison when types and overflow allow" in fix
    assert "raw column amount" in fix
    assert "1 + 2 / 2.0" not in fix


@pytest.mark.parametrize("dialect", DIALECTS)
def test_gv005_nonnumeric_comparison_constants_do_not_get_invalid_arithmetic_advice(dialect):
    # amount + 5 = '20' is not a safe numeric rewrite target; advice must stay general and not emit broken SQL.
    fix = gv005_fix("select id from t where amount + 5 = '20'", dialect)
    assert "Move the constant arithmetic to the other side of the comparison when types and overflow allow" in fix
    assert "raw column amount" in fix
    assert "amount = '20' - 5" not in fix
    con = duckdb.connect()
    con.execute("create table t(id int, amount int)")
    con.execute("insert into t values (1, 15)")
    # Demonstrate the broken rewrite would raise; we must not suggest it.
    with pytest.raises(duckdb.BinderException):
        con.execute("select id from t where amount = '20' - 5")


@pytest.mark.parametrize("dialect", DIALECTS)
def test_gv005_coalesce_does_not_claim_equivalence_for_coerced_or_null_cases(dialect):
    # For DOUBLE n containing NULL, COALESCE(n, 1) = 1.00000000000000001 can be true after SQL coercion,
    # while dropping/rewriting COALESCE based on literal equality is unsafe without schema knowledge.
    fix = gv005_fix("select id from t where coalesce(n, 1) = 1.00000000000000001", dialect)
    assert "Compare the raw column n directly and handle NULL explicitly" in fix
    assert "Do not drop COALESCE" in fix
    assert "n = 1.00000000000000001" not in fix
    assert "OR n IS NULL" not in fix
    assert "AND n IS NOT NULL" not in fix
    con = duckdb.connect()
    con.execute("create table t(id int, n double)")
    con.execute("insert into t values (1, 1.0), (2, NULL), (3, 2.0)")
    original = "select id from t where coalesce(n, 1) = 1.00000000000000001 order by id"
    # SQL coercion: NULL -> 1, compared as DOUBLE 1.0 = 1.00000000000000001 -> true.
    assert rows(con, original) == [(1,), (2,)]
    # A literal-equality check (Decimal(1) != Decimal(1.00000000000000001)) would wrongly add IS NOT NULL
    # and drop the NULL row, so no concrete rewrite may be claimed.
    unsafe = "select id from t where n = 1.00000000000000001 and n is not null order by id"
    assert rows(con, unsafe) == [(1,)]
    assert rows(con, original) != rows(con, unsafe)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_gv005_coalesce_type_promotion_and_affinity_are_not_claimed_equivalent(dialect):
    # COALESCE(n, 0e0) = 9007199254740993 promotes BIGINT to DOUBLE semantics; removing COALESCE changes behavior.
    fix_double = gv005_fix("select id from t where coalesce(n, 0e0) = 9007199254740993", dialect)
    assert "Compare the raw column n directly and handle NULL explicitly" in fix_double
    assert "Do not drop COALESCE" in fix_double
    con = duckdb.connect()
    con.execute("create table t(id int, n bigint)")
    con.execute("insert into t values (1, 9007199254740992)")
    original = "select id from t where coalesce(n, 0e0) = 9007199254740993 order by id"
    naive_drop = "select id from t where n = 9007199254740993 order by id"
    # COALESCE with the DOUBLE default promotes 9007199254740992 to DOUBLE, where it equals the
    # rounded literal; the raw BIGINT comparison is false. Removing COALESCE changes the result.
    assert rows(con, original) == [(1,)]
    assert rows(con, naive_drop) == []
    assert rows(con, original) != rows(con, naive_drop)

    # SQLite INTEGER affinity / text-vs-integer coercion example: COALESCE(n, '0') = '0' differs when removed.
    fix_text = gv005_fix("select id from t where coalesce(n, '0') = '0'", dialect)
    assert "Compare the raw column n directly and handle NULL explicitly" in fix_text
    sqlite = sqlite3.connect(":memory:")
    sqlite.execute("create table t2(id int, n integer)")
    sqlite.executemany("insert into t2 values (?, ?)", [(1, None), (2, 0), (3, 5)])
    original_text = "select id from t2 where coalesce(n, '0') = '0' order by id"
    dropped = "select id from t2 where n = '0' order by id"
    # The NULL row matches after coercion by the default; dropping COALESCE matches the INTEGER 0 instead.
    assert rows(sqlite, original_text) == [(1,)]
    assert rows(sqlite, dropped) == [(2,)]


@pytest.mark.parametrize("dialect", FORMAT_DIALECTS)
def test_gv005_numeric_and_quoted_and_unknown_tochar_formats_do_not_get_date_advice(dialect):
    # Numeric masks stay numeric when Postgres normalizes D/MI to %u/%M next to digits.
    for fmt in ("FM999MI", "999D99", "999TH", "990", "999.00", "%M99", "%u99", "%OM99", "%Ou99"):
        numeric_fix = gv005_fix(f"select id from t where to_char(amt, '{fmt}') = '123'", dialect)
        assert "constant range" not in numeric_fix, (dialect, fmt, numeric_fix)
        assert "DATE(" not in numeric_fix
        assert "2024-01-01" not in numeric_fix
        assert "Compare the raw column amt where possible" in numeric_fix

    # YYYY inside a quoted literal span is not a temporal token.
    quoted_fix = gv005_fix("select id from t where to_char(amount, 'FM999\"YYYY\"') = '123'", dialect)
    assert "constant range" not in quoted_fix
    assert "DATE(" not in quoted_fix
    assert "2024-01-01" not in quoted_fix
    assert "Compare the raw column amount where possible" in quoted_fix

    # Temporal fields, compact field sequences, literal zeros, and ordinal TH keep date advice.
    for fmt in ("YYYY-MM-DD 00:00", "DDTH", "DDTH Month YYYY", "YYYYMMDD", "HH24MISS", "YYYYMMDDHH24MISS"):
        fix_tochar = gv005_fix(f"select id from t where to_char(ts, '{fmt}') = 'x'", dialect)
        assert "constant range" in fix_tochar and "DATE(ts)" in fix_tochar, (dialect, fmt, fix_tochar)

    # Unknown formats must not receive date advice.
    unknown_fix = gv005_fix("select id from t where to_char(x, 'JSON') = '1'", dialect)
    assert "constant range" not in unknown_fix
    assert "DATE(" not in unknown_fix
    assert "2024-01-01" not in unknown_fix
    assert "Compare the raw column x where possible" in unknown_fix


@pytest.mark.parametrize("dialect", FORMAT_DIALECTS)
def test_gv005_timetostr_supports_valid_strftime_directives_including_abbreviations(dialect):
    # %a and %B are valid strftime directives and should be recognized as temporal.
    fix_a = gv005_fix("select id from t where strftime(ts, '%a') = 'Mon'", dialect)
    assert fix_a == (
        "Compare the raw column to a constant range, e.g. ts >= '2024-01-01' AND "
        "ts < '2024-01-02' instead of DATE(ts) = '2024-01-01'."
    )
    fix_b = gv005_fix("select id from t where strftime(ts, '%B') = 'January'", dialect)
    assert fix_b == (
        "Compare the raw column to a constant range, e.g. ts >= '2024-01-01' AND "
        "ts < '2024-01-02' instead of DATE(ts) = '2024-01-01'."
    )
    # Common temporal directives, literal 0/9 digits, and padding flags remain temporal.
    for expr in (
        "strftime(ts, '%Y-%m-%d') = '2024-01-01'",
        "strftime(ts, '%Y-%m-%d 00:00') = '2024-01-01 00:00'",
        "strftime(ts, '%-d') = '1'",
        "strftime(ts, '%M99') = '123'",
        "strftime(ts, '%u99') = '123'",
        "format_timestamp('%Y-%m-%d 00:00', ts) = '2024-01-01 00:00'",
    ):
        fix_t = gv005_fix(f"select id from t where {expr}", dialect)
        assert "constant range" in fix_t and "DATE(ts)" in fix_t, (dialect, expr, fix_t)


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("where, expected_copy, unexpected_case", [
    ("upper(name) like 'A%'", "UPPER(name)", "lower-case"),
    ("upper(name) like 'I%'", "UPPER(name)", "lower-case"),
    ("upper(name) = 'A_B'", "UPPER(name)", "lower-case"),
    ("upper(name) > 'M'", "UPPER(name)", "lower-case"),
    ("lower(name) like 'A%'", "LOWER(name)", "upper-case"),
    ("lower(name) like 'a%'", "LOWER(name)", "upper-case"),
    ("'ALICE' like upper(pattern)", "UPPER(pattern)", "lower-case"),
    ("'alice' like lower(pattern)", "LOWER(pattern)", "upper-case"),
])
def test_gv005_upper_and_lower_recommend_exact_precomputed_copy_without_ilike(
    dialect, where, expected_copy, unexpected_case
):
    fix = gv005_fix(f"select id from t where {where} limit 5", dialect)
    assert "ILIKE" not in fix
    assert expected_copy in fix
    assert unexpected_case not in fix
    assert "without changing the comparison operator or operand order" in fix


@pytest.mark.parametrize("dialect", FORMAT_DIALECTS)
@pytest.mark.parametrize("fmt", ["YYYYFOO", "FOOYYYY"])
def test_gv005_partial_unknown_tochar_formats_do_not_get_date_advice(dialect, fmt):
    fix = gv005_fix(f"select id from t where to_char(x, '{fmt}') = '2024FOO'", dialect)
    assert "constant range" not in fix
    assert "DATE(" not in fix
    assert "2024-01-01" not in fix
    assert fix == ("Compare the raw column x where possible, or filter on a partition or indexed column first "
                   "so less data is read.")


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("where", ["upper(trim(name)) like 'ALICE'", "trim(lower(name)) = 'alice'"])
def test_gv005_nested_case_or_trim_does_not_drop_inner_operation(dialect, where):
    fix = gv005_fix(f"select id from t where {where} limit 5", dialect)
    assert "ILIKE" not in fix
    assert "precomputed lower-case" not in fix
    assert "precomputed upper-case" not in fix
    assert "Store name trimmed" not in fix
    assert "transformed value" in fix
    assert "collation and type-affinity semantics" in fix
    assert "without type affinity" in fix
    assert "without changing the comparison operator or operand order" in fix


def test_gv005_duckdb_upper_like_and_ordering_counterexamples():
    con = duckdb.connect()
    con.execute("create table t(id int, name varchar, pattern varchar)")
    con.executemany(
        "insert into t values (?, ?, ?)",
        [(1, "ı", "a%"), (2, "a", "b%"), (3, "_", "ALICE"), (4, "Alice", "c%")],
    )
    # U+0131 (dotless i): UPPER('ı') == 'I' matches LIKE 'I%', while ILIKE compares 'ı' vs 'i' and misses it.
    assert rows(con, "select id from t where UPPER(name) LIKE 'I%' order by id") == [(1,)]
    assert rows(con, "select id from t where name ILIKE 'I%' order by id") == []
    # Ordered comparison: UPPER(name) > 'M' rejects 'a' and matches '_', whereas LOWER(name) > 'm' flips both.
    assert rows(con, "select id from t where UPPER(name) > 'M' order by id") == [(3,)]
    assert rows(con, "select id from t where LOWER(name) > 'm' order by id") == [(1,)]
    assert rows(con, "select id from t where name > 'M' order by id") == [(1,), (2,), (3,)]
    # Reversing pattern operand ('ALICE' LIKE UPPER(pattern)) also changes matching rows.
    assert rows(con, "select id from t where 'ALICE' LIKE UPPER(pattern) order by id") == [(1,), (3,)]
    assert rows(con, "select id from t where pattern ILIKE 'ALICE' order by id") == [(3,)]
@pytest.mark.parametrize("dialect, sql_expr, expected_token", [
    ("postgres", "trim(leading '0' from code)", "LEADING '0'"),
    ("duckdb", "trim(leading '0' from code)", "LTRIM(code, '0')"),
    ("sqlite", "trim(leading '0' from code)", "LTRIM(code, '0')"),
    ("snowflake", "trim(code, '0')", "TRIM(code, '0')"),
])
def test_gv005_trim_with_options_preserves_trim_chars_and_direction(dialect, sql_expr, expected_token):
    fix = gv005_fix(f"select id from t where {sql_expr} = '1' limit 5", dialect)
    assert "TRIM(code) runs on every row" not in fix
    assert expected_token in fix
    assert "collation and type-affinity semantics" in fix


def test_gv005_sqlite_affinity_and_trim_options_counterexamples():
    # SQLite affinity: UPPER(name) = 1 is false for name = '1', whereas UPPER_COPY TEXT coerces 1 to '1'.
    f = gv005_fix("select id from t where upper(name) = 1 limit 5", "sqlite")
    assert "without type affinity" in f
    con_sqlite = sqlite3.connect(":memory:")
    con_sqlite.execute("create table t(id int, name text, name_upper_text text, name_upper_none)")
    con_sqlite.execute("insert into t values (1, '1', '1', '1')")
    assert rows(con_sqlite, "select id from t where UPPER(name) = 1") == []
    assert rows(con_sqlite, "select id from t where name_upper_text = 1") == [(1,)]
    assert rows(con_sqlite, "select id from t where name_upper_none = 1") == []
    # Trim options: TRIM(LEADING '0' FROM code) = '1' differs from TRIM(code) = '1' on '001'.
    con_duck = duckdb.connect()
    con_duck.execute("create table t(id int, code varchar)")
    con_duck.execute("insert into t values (1, '001')")
    assert rows(con_duck, "select id from t where TRIM(LEADING '0' FROM code) = '1'") == [(1,)]
    assert rows(con_duck, "select id from t where TRIM(code) = '1'") == []
