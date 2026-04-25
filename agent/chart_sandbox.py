"""Sandboxed execution of LLM-generated chart code.

The LLM writes Python that produces a `fig` variable. We:

1. **AST-validate** the code — reject imports, dunder access, and a denylist
   of dangerous names. Static check, fast.
2. **Restrict globals** — the sandbox only sees `df`, `pd`, `px`, `go`, `np`,
   and a curated set of safe builtins. No `open`, `eval`, `exec`,
   `__import__`, `globals`, `locals`, etc.
3. **Timeout** via thread-pool — code that loops is killed (soft kill; the
   thread remains running in the background, but the main flow continues).
4. **Type-check the output** — `fig` must be a `plotly.graph_objects.Figure`
   (which `px.bar/line/pie/...` also returns).

This is **demo-grade isolation**, not production-grade. Determined sandbox
escapes via Python introspection tricks ([].__class__.__bases__[0]....) are
blocked by the AST dunder check, but a creative attacker has many wedges.
For public-internet deployment, swap in a true isolation layer (Pyodide,
seccomp'd subprocess, container per-exec). For trusted-LLM portfolio demo,
this is a reasonable trade-off — much more flexibility for the model than
a rigid ChartSpec schema, with low risk under the current threat model.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _ExecTimeout
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SANDBOX_TIMEOUT_SECONDS = 5.0

# Names the AST walker rejects on sight — escapes from the sandbox or
# clearly malicious operations. Curated allowlist for builtins is below.
DISALLOWED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit", "help",
    "__builtins__", "__loader__", "__spec__", "__file__",
    "__name__", "__class__", "__bases__", "__subclasses__",
    "__init__", "__new__", "__del__",
}

# Builtins the sandbox is allowed to use. Anything not on this list
# (open, eval, etc.) is unavailable inside exec().
ALLOWED_BUILTINS: dict[str, Any] = {
    # Math / aggregation
    "len": len, "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "pow": pow, "divmod": divmod,
    # Iteration / sorting
    "sorted": sorted, "reversed": reversed, "any": any, "all": all,
    "enumerate": enumerate, "zip": zip, "range": range, "filter": filter,
    "map": map, "iter": iter, "next": next, "slice": slice,
    # Type constructors / coercions
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    "str": str, "int": int, "float": float, "bool": bool, "complex": complex,
    "bytes": bytes,
    # Type checks (read-only — `type` is fine; AST already blocks dunder access
    # so `type(x).__bases__` etc. is unreachable).
    "isinstance": isinstance, "issubclass": issubclass, "type": type,
    "hasattr": hasattr,  # AST blocks dunder access; `getattr` stays denied
    # because `getattr(obj, '__class__')` bypasses the AST's literal check.
    # Constants
    "True": True, "False": False, "None": None,
    # Errors (so the LLM can raise/catch)
    "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
    "KeyError": KeyError, "IndexError": IndexError,
    # Misc
    "print": print,  # harmless; output goes to logs
    "repr": repr, "format": format,
}


class SandboxError(Exception):
    """Raised on any sandbox failure — AST violation, runtime error, timeout."""


def validate_ast(code: str) -> None:
    """Reject code that could escape the sandbox via static patterns."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"Syntax error: {e.msg} (line {e.lineno})")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("Import statements are not allowed")
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            raise SandboxError(f"Disallowed name: '{node.id}'")
        if isinstance(node, ast.Attribute):
            # Block dunder attribute access — main escape vector
            # (e.g., obj.__class__.__bases__[0].__subclasses__()).
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SandboxError(f"Dunder attribute access not allowed: '{node.attr}'")


# Single shared executor — chart execs are bounded by SANDBOX_TIMEOUT_SECONDS,
# so a small pool is fine even under burst load.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chart_sandbox")


def execute_chart_code(code: str, rows: list[dict]) -> go.Figure | None:
    """Run LLM-generated chart code against `rows`, return the produced `fig`.

    Raises SandboxError on validation/runtime failure. The caller (typically
    `_build_chart`) is expected to catch and degrade gracefully to "no chart"
    rather than killing the whole turn.
    """
    validate_ast(code)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    sandbox_globals: dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "df": df,
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
    }
    sandbox_locals: dict[str, Any] = {}

    def _run():
        exec(code, sandbox_globals, sandbox_locals)
        return sandbox_locals.get("fig", sandbox_globals.get("fig"))

    try:
        fig = _executor.submit(_run).result(timeout=SANDBOX_TIMEOUT_SECONDS)
    except _ExecTimeout:
        raise SandboxError(
            f"Chart code timed out after {SANDBOX_TIMEOUT_SECONDS}s "
            "(thread continues running in background; main flow proceeds)"
        )
    except SandboxError:
        raise
    except Exception as e:
        raise SandboxError(f"Runtime error: {type(e).__name__}: {e}")

    if fig is None:
        return None
    if not isinstance(fig, go.Figure):
        raise SandboxError(
            f"`fig` must be a plotly Figure (px.bar/line/pie/... or go.Figure); "
            f"got {type(fig).__name__}"
        )
    return fig
