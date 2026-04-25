import pytest

from agent.sql_guard import DEFAULT_LIMIT, guard


# ---------- happy paths ----------

def test_simple_select_passes_and_injects_limit():
    out = guard("SELECT 1")
    assert "LIMIT" in out.upper()
    assert str(DEFAULT_LIMIT) in out


def test_explicit_limit_preserved():
    out = guard("SELECT 1 LIMIT 5")
    assert "LIMIT 5" in out
    assert str(DEFAULT_LIMIT) not in out


def test_select_with_filter_and_orderby():
    out = guard("SELECT name FROM customer WHERE active = 1 ORDER BY customer_id")
    assert "WHERE" in out.upper()
    assert "ORDER BY" in out.upper()
    assert str(DEFAULT_LIMIT) in out


def test_with_cte_select_injects_limit_on_outer():
    sql = (
        "WITH active_c AS (SELECT customer_id FROM customer WHERE active = 1) "
        "SELECT count(*) FROM active_c"
    )
    out = guard(sql)
    assert "WITH" in out.upper()
    assert str(DEFAULT_LIMIT) in out


def test_union_select_passes():
    out = guard("SELECT 1 UNION SELECT 2")
    assert "UNION" in out.upper()
    assert str(DEFAULT_LIMIT) in out


def test_trailing_semicolon_stripped():
    out = guard("SELECT 1;")
    assert "LIMIT" in out.upper()


def test_complex_real_query():
    sql = (
        "SELECT f.title, COUNT(r.rental_id) AS rental_count "
        "FROM rental r JOIN inventory i ON r.inventory_id = i.inventory_id "
        "JOIN film f ON i.film_id = f.film_id "
        "WHERE r.rental_date >= '2022-08-01' AND r.rental_date < '2022-09-01' "
        "GROUP BY f.title ORDER BY rental_count DESC LIMIT 5"
    )
    out = guard(sql)
    assert "LIMIT 5" in out
    assert str(DEFAULT_LIMIT) not in out


# ---------- mutation rejections ----------

def test_insert_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("INSERT INTO customer (first_name) VALUES ('x')")


def test_update_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("UPDATE customer SET first_name = 'x'")


def test_delete_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("DELETE FROM customer")


def test_drop_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("DROP TABLE customer")


def test_create_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("CREATE TABLE foo (id int)")


def test_truncate_rejected():
    with pytest.raises(ValueError, match=r"(?i)select|mutation"):
        guard("TRUNCATE TABLE customer")


# ---------- mutation hidden inside CTE ----------

def test_delete_in_cte_rejected():
    sql = "WITH del AS (DELETE FROM customer RETURNING customer_id) SELECT * FROM del"
    with pytest.raises(ValueError, match=r"(?i)mutation"):
        guard(sql)


def test_insert_in_cte_rejected():
    sql = (
        "WITH ins AS (INSERT INTO customer (first_name) VALUES ('x') RETURNING customer_id) "
        "SELECT * FROM ins"
    )
    with pytest.raises(ValueError, match=r"(?i)mutation"):
        guard(sql)


def test_update_in_cte_rejected():
    sql = (
        "WITH upd AS (UPDATE customer SET first_name = 'x' RETURNING customer_id) "
        "SELECT * FROM upd"
    )
    with pytest.raises(ValueError, match=r"(?i)mutation"):
        guard(sql)


# ---------- multi-statement / parse failures ----------

def test_multiple_statements_rejected():
    with pytest.raises(ValueError, match=r"(?i)one"):
        guard("SELECT 1; SELECT 2")


def test_invalid_sql_rejected():
    with pytest.raises(ValueError):
        guard("SELECT FROM WHERE bogus")


def test_empty_string_rejected():
    with pytest.raises(ValueError, match=r"(?i)empty"):
        guard("")


def test_whitespace_only_rejected():
    with pytest.raises(ValueError, match=r"(?i)empty"):
        guard("   \n\t  ")


# ---------- dangerous functions ----------

def test_pg_read_file_rejected():
    with pytest.raises(ValueError, match=r"(?i)pg_read_file"):
        guard("SELECT pg_read_file('/etc/passwd')")


def test_lo_import_rejected():
    with pytest.raises(ValueError, match=r"(?i)lo_import"):
        guard("SELECT lo_import('/etc/passwd')")


def test_dblink_rejected():
    with pytest.raises(ValueError, match=r"(?i)dblink"):
        guard("SELECT * FROM dblink('host=evil', 'SELECT 1') AS t(x int)")


# ---------- benign functions still pass ----------

def test_safe_aggregates_pass():
    out = guard("SELECT count(*), sum(amount), avg(amount) FROM payment")
    assert "count" in out.lower()
    assert str(DEFAULT_LIMIT) in out


def test_string_functions_pass():
    out = guard("SELECT lower(first_name), upper(last_name) FROM customer")
    assert "lower" in out.lower()
