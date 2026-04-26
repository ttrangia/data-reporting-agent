"""Eval runner — loads dataset.yaml, runs each case through the live
LangGraph agent, applies predicates, and writes a markdown scorecard.

Hits real Postgres + real Anthropic. The point of this harness is exactly
what unit tests can't measure: end-to-end behavior under the model the
user actually talks to. Costs money on every run; budget for ~$0.50/run
at ~12 cases.

Usage:
    python -m evals.runner                   # full run
    python -m evals.runner --id top_films    # one case
    python -m evals.runner --limit 3         # first N cases
    python -m evals.runner --concurrency 1   # serial (easier to debug)
    python -m evals.runner --no-report       # skip writing the scorecard
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()  # must come before anything that reads DATABASE_URL_AGENT etc.

from agent.graph import app_graph  # noqa: E402
from agent.state import turn_input  # noqa: E402
from evals import predicates as P  # noqa: E402
from evals.scorecard import write_scorecard  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evals" / "dataset.yaml"
REPORTS_DIR = REPO_ROOT / "evals" / "reports"


@dataclass
class TurnResult:
    question: str
    state: dict
    error: str | None = None  # set if the graph raised


@dataclass
class CaseResult:
    case_id: str
    notes: str | None
    turn_results: list[TurnResult]
    predicate_results: list[P.PredicateResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def crashed(self) -> bool:
        return any(t.error for t in self.turn_results)

    @property
    def passed(self) -> bool:
        if self.crashed:
            return False
        if not self.predicate_results:
            return False  # a case with no checks isn't a meaningful pass
        return all(p.passed for p in self.predicate_results)


def _load_cases(path: Path = DATASET_PATH) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list at top of {path}, got {type(raw).__name__}")
    return raw


def _normalize_case(case: dict) -> dict:
    """Single-turn shortcut: question+expected → one-element turns list."""
    if "turns" in case:
        return case
    if "question" not in case:
        raise ValueError(f"Case {case.get('id')!r} has neither 'turns' nor 'question'.")
    return {
        "id": case["id"],
        "notes": case.get("notes"),
        "turns": [{"question": case["question"], "expected": case.get("expected", {})}],
    }


def _apply_predicates(expected: dict, state: dict) -> list[P.PredicateResult]:
    out: list[P.PredicateResult] = []
    for name, arg in (expected or {}).items():
        try:
            fn = P.get(name)
            out.append(fn(arg, state))
        except Exception as e:  # noqa: BLE001 — predicate bugs should fail the case loudly
            out.append(P.PredicateResult(name, False, f"predicate raised: {type(e).__name__}: {e}"))
    return out


async def _run_case(case: dict, run_tag: str | None = None) -> CaseResult:
    case_id = case["id"]
    notes = case.get("notes")
    thread_id = f"eval-{case_id}-{uuid.uuid4().hex[:8]}"

    turn_results: list[TurnResult] = []
    pred_results: list[P.PredicateResult] = []

    started = time.perf_counter()
    for turn_idx, turn in enumerate(case["turns"], 1):
        question = turn["question"]
        expected = turn.get("expected") or {}
        # LangSmith tags + metadata. Tagging "eval" + the case_id lets you
        # filter "show me failed cases this week" entirely in the LangSmith
        # UI. run_tag is freeform — pass --tag baseline / --tag after-rag
        # to label runs you want to compare side-by-side.
        tags = ["eval", f"case:{case_id}"]
        if run_tag:
            tags.append(f"run:{run_tag}")
        config = {
            "configurable": {"thread_id": thread_id},
            "tags": tags,
            "metadata": {
                "source": "eval",
                "case_id": case_id,
                "turn_index": turn_idx,
                "run_tag": run_tag,
            },
            "run_name": f"eval:{case_id}" + (f":turn{turn_idx}" if len(case["turns"]) > 1 else ""),
        }
        try:
            state = await app_graph.ainvoke(
                turn_input(question, HumanMessage(content=question)),
                config=config,
            )
            turn_results.append(TurnResult(question=question, state=state))
            # Offload to a worker thread — the LLM-judge predicate makes a
            # blocking API call, and we don't want that to stall sibling
            # cases running concurrently in the same event loop.
            pred_results.extend(
                await asyncio.to_thread(_apply_predicates, expected, state)
            )
        except Exception as e:  # noqa: BLE001 — graph failures shouldn't kill the whole run
            turn_results.append(
                TurnResult(question=question, state={}, error=f"{type(e).__name__}: {e}")
            )
            break  # don't run subsequent turns of a crashed case
    elapsed = time.perf_counter() - started

    return CaseResult(
        case_id=case_id,
        notes=notes,
        turn_results=turn_results,
        predicate_results=pred_results,
        elapsed_s=elapsed,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip() or "no-git"
    except Exception:
        return "no-git"


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--id", help="Run a single case by id.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Cases to run in parallel (default 3). Use 1 for easier debugging.",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip writing the scorecard markdown file.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Freeform label attached to each LangSmith trace as a tag "
             "(e.g. 'baseline', 'after-rag'). Useful for side-by-side "
             "comparison in the LangSmith dashboard.",
    )
    args = parser.parse_args()

    cases_raw = _load_cases()
    cases = [_normalize_case(c) for c in cases_raw]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"No case with id={args.id!r}.")
            return 1
    if args.limit:
        cases = cases[: args.limit]

    # Surface tracing status so the user knows whether traces will land in
    # LangSmith for this run. Cheap sanity check before burning API credits.
    tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
    project = os.getenv("LANGSMITH_PROJECT") or "(default)"
    if tracing_on:
        print(f"LangSmith tracing: ON · project={project}"
              + (f" · run_tag={args.tag}" if args.tag else ""), flush=True)
    else:
        print("LangSmith tracing: off (set LANGSMITH_TRACING=true to enable)", flush=True)

    print(f"Running {len(cases)} case(s) at concurrency={args.concurrency}…", flush=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(case):
        async with sem:
            res = await _run_case(case, run_tag=args.tag)
            mark = "✓" if res.passed else ("💥" if res.crashed else "✗")
            print(f"  {mark}  {res.case_id} ({res.elapsed_s:.1f}s)", flush=True)
            return res

    started = time.perf_counter()
    results = await asyncio.gather(*[_bounded(c) for c in cases])
    total_elapsed = time.perf_counter() - started

    # Preserve dataset order for the scorecard (gather may interleave)
    order = {c["id"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.case_id, 1_000_000))

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} passed · {total_elapsed:.1f}s total")

    if not args.no_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sha = _git_sha()
        ts = time.strftime("%Y-%m-%d_%H%M")
        out_path = REPORTS_DIR / f"{ts}_{sha}.md"
        write_scorecard(out_path, results, total_elapsed_s=total_elapsed, git_sha=sha)
        print(f"Scorecard: {out_path.relative_to(REPO_ROOT)}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
