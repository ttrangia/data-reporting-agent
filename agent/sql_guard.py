"""SQL validator — fence 2 of the read-only guarantee.

Parses with sqlglot, asserts top-level is SELECT/UNION (optionally inside WITH),
walks the entire AST rejecting mutation nodes anywhere (catches DML inside CTEs:
`WITH del AS (DELETE FROM x RETURNING *) SELECT * FROM del` is a real Postgres
feature), rejects multiple statements, rejects dangerous functions, and injects
a default LIMIT when one is not supplied.

Raises ValueError with a message worded for an LLM retry: state the violation
specifically so the SQL generator can fix it.
"""
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DEFAULT_LIMIT = 1000

# Mutation node types — anywhere these appear in the AST is a hard reject.
MUTATION_NODES: tuple = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.AlterColumn,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
)

# Functions with filesystem, network, or escalation reach. Not exhaustive —
# defense in depth, paired with a non-superuser DB role.
DANGEROUS_FUNCTIONS = frozenset({
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "lo_create",
    "lo_unlink",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    "dblink_send_query",
    "copy_from",
    "copy_to",
})


def _func_name(node: exp.Expression) -> str:
    """Best-effort lowercase function name from sqlglot Func or Anonymous nodes."""
    if isinstance(node, exp.Anonymous):
        # Anonymous.this is the raw function name string
        raw = node.this
        return (raw or "").lower() if isinstance(raw, str) else ""
    if isinstance(node, exp.Func):
        # Built-in funcs have .name set
        try:
            return (node.name or "").lower()
        except Exception:
            return ""
    return ""


def guard(sql: str) -> str:
    """Validate and canonicalize SQL. Returns canonical SQL with LIMIT injected.

    Raises ValueError on any policy violation. The error message is intended
    for inclusion in retry prompts, so it names the specific problem.
    """
    if not sql or not sql.strip():
        raise ValueError("Empty SQL.")

    cleaned = sql.strip().rstrip(";").strip()

    try:
        statements = sqlglot.parse(cleaned, dialect="postgres")
    except ParseError as e:
        raise ValueError(f"SQL failed to parse: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        raise ValueError("SQL parsed to zero statements.")
    if len(statements) > 1:
        raise ValueError(
            f"Only one SQL statement allowed; received {len(statements)}. "
            "Combine into a single query."
        )

    tree = statements[0]

    # Top-level: must be SELECT or UNION (optionally wrapped in WITH for CTEs).
    top = tree.this if isinstance(tree, exp.With) else tree
    if not isinstance(top, (exp.Select, exp.Union)):
        raise ValueError(
            f"Top-level statement must be SELECT or UNION; got "
            f"{type(top).__name__}. Read-only queries only."
        )

    # Walk the WHOLE tree (including CTE bodies) for mutation nodes.
    for mut in tree.find_all(*MUTATION_NODES):
        raise ValueError(
            f"Mutation operation '{type(mut).__name__}' is not allowed anywhere "
            "in the query, including CTEs. Only SELECT queries are permitted."
        )

    # Walk for dangerous functions (filesystem / network / escalation).
    for fn in tree.find_all(exp.Func, exp.Anonymous):
        name = _func_name(fn)
        if name in DANGEROUS_FUNCTIONS:
            raise ValueError(
                f"Function '{name}' is not allowed (filesystem/network/escalation risk). "
                "Use only standard query functions."
            )

    # Inject LIMIT on the outermost SELECT / UNION if one isn't already set.
    limit_target = tree.this if isinstance(tree, exp.With) else tree
    if isinstance(limit_target, (exp.Select, exp.Union)) and not limit_target.args.get("limit"):
        limit_target.set(
            "limit",
            exp.Limit(expression=exp.Literal.number(DEFAULT_LIMIT)),
        )

    return tree.sql(dialect="postgres")
