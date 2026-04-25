"""Tests for the chart code sandbox — AST validation + restricted exec."""
import pytest
import plotly.graph_objects as go

from agent.chart_sandbox import (
    SandboxError,
    execute_chart_code,
    validate_ast,
)


# ---------- AST validation: rejection cases ----------

def test_imports_rejected():
    with pytest.raises(SandboxError, match="Import"):
        validate_ast("import os\nfig = None")


def test_from_imports_rejected():
    with pytest.raises(SandboxError, match="Import"):
        validate_ast("from os import system\nfig = None")


def test_eval_rejected():
    with pytest.raises(SandboxError, match="eval"):
        validate_ast("fig = eval('1+1')")


def test_exec_rejected():
    with pytest.raises(SandboxError, match="exec"):
        validate_ast("exec('fig = None')")


def test_open_rejected():
    with pytest.raises(SandboxError, match="open"):
        validate_ast("f = open('/etc/passwd')\nfig = None")


def test_dunder_class_access_rejected():
    """The classic sandbox-escape: obj.__class__.__bases__[0].__subclasses__()."""
    with pytest.raises(SandboxError, match="Dunder"):
        validate_ast("x = (1).__class__\nfig = None")


def test_dunder_subclasses_rejected():
    with pytest.raises(SandboxError, match="Dunder"):
        validate_ast("x = ().__subclasses__()\nfig = None")


def test_syntax_error_caught():
    with pytest.raises(SandboxError, match="Syntax"):
        validate_ast("fig = 1 +")


# ---------- AST validation: clean code passes ----------

def test_valid_chart_code_passes_ast():
    code = """
top = df.nlargest(5, 'count')
fig = px.bar(top, x='title', y='count', title='Top 5')
fig.update_layout(yaxis_title='Rentals')
"""
    validate_ast(code)  # no exception


def test_valid_with_pandas_transformations():
    code = """
df['pct'] = df['revenue'] / df['revenue'].sum()
top = df.sort_values('pct', ascending=False).head(10)
fig = px.bar(top, x='name', y='pct', title='Top 10 by share')
fig.update_yaxes(tickformat='.1%')
"""
    validate_ast(code)


# ---------- execute_chart_code: happy paths ----------

def test_execute_returns_figure():
    rows = [{"title": "A", "count": 5}, {"title": "B", "count": 3}]
    code = "fig = px.bar(df, x='title', y='count', title='Test')"
    fig = execute_chart_code(code, rows)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test"


def test_execute_with_transformation():
    rows = [{"name": "X", "v": 10}, {"name": "Y", "v": 5}, {"name": "Z", "v": 20}]
    code = """
top = df.nlargest(2, 'v')
fig = px.bar(top, x='name', y='v', title='Top 2')
"""
    fig = execute_chart_code(code, rows)
    assert isinstance(fig, go.Figure)
    # Bar chart trace — should reflect the nlargest(2) transformation
    assert len(fig.data[0].x) == 2


def test_execute_fig_none_returns_none():
    """LLM declines to chart — sandbox returns None cleanly."""
    rows = [{"x": 1}]
    code = "fig = None"
    assert execute_chart_code(code, rows) is None


def test_execute_with_no_rows_returns_none():
    """Empty rows → None without executing user code (defensive)."""
    code = "fig = px.bar(df, x='a', y='b')"
    assert execute_chart_code(code, []) is None


# ---------- execute_chart_code: failure modes ----------

def test_runtime_error_wrapped_as_sandbox_error():
    rows = [{"a": 1}]
    code = "fig = px.bar(df, x='nonexistent_column', y='a')"
    with pytest.raises(SandboxError, match="Runtime"):
        execute_chart_code(code, rows)


def test_wrong_return_type_rejected():
    rows = [{"a": 1}]
    code = "fig = 'not a figure'"
    with pytest.raises(SandboxError, match="Figure"):
        execute_chart_code(code, rows)


def test_ast_violation_caught_before_execution():
    rows = [{"a": 1}]
    code = "import os\nfig = None"
    with pytest.raises(SandboxError, match="Import"):
        execute_chart_code(code, rows)


# ---------- sandbox isolation ----------

def test_no_access_to_open_via_globals():
    """Even though `open` exists in normal Python, the sandbox doesn't expose it."""
    rows = [{"a": 1}]
    # Use a NON-dunder reference so AST validation doesn't catch it first;
    # fail at runtime due to NameError (open not in restricted builtins).
    code = "f = open\nfig = None"
    with pytest.raises(SandboxError):
        execute_chart_code(code, rows)


def test_attempting_to_assign_globals_fails():
    """Direct __builtins__ access is blocked by AST."""
    rows = [{"a": 1}]
    code = "__builtins__['open'] = open\nfig = None"
    with pytest.raises(SandboxError):
        execute_chart_code(code, rows)
