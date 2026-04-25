"""One-shot script to measure how many tokens each prompt component consumes
on Sonnet 4.5. Uses Anthropic's `messages.count_tokens` API (free, exact).

Run with:
    venv/bin/python -m tools.measure_tokens
"""
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from agent.db import pagila_schema_string
from agent.prompts import (
    CHART_FORCE_NOTE,
    CHART_SPEC_SYSTEM,
    DATASET_NOTES,
    FRONT_AGENT_SYSTEM,
    SQL_GENERATION_RETRY_HINT,
    SQL_GENERATION_SYSTEM,
    SUMMARIZE_SYSTEM,
)

# Pricing for Sonnet 4.5 — keep current with Anthropic's published rates.
# Input  $3 / 1M tokens
# Output $15 / 1M tokens
PRICE_PER_M_INPUT = 3.00
MODEL = "claude-sonnet-4-5"

client = Anthropic()


def n_tokens(text: str) -> int:
    """Count input tokens for a single user message containing `text`.
    count_tokens is a free endpoint — does not run inference."""
    resp = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def cost_per_1k_calls(tokens: int) -> str:
    """Dollar cost if this prompt is sent 1k times (input only, no caching)."""
    dollars = tokens * 1000 * (PRICE_PER_M_INPUT / 1_000_000)
    return f"${dollars:>6.2f}"


def main() -> None:
    schema = pagila_schema_string()

    print(f"\nModel: {MODEL}")
    print(f"Pricing assumed: ${PRICE_PER_M_INPUT}/M input tokens")
    print()

    # --- Component-level (raw chunks) ---
    components = [
        ("Pagila schema (DDL + 3 sample rows × 22 tables)", schema),
        ("DATASET_NOTES",                                    DATASET_NOTES),
        ("FRONT_AGENT_SYSTEM template",                      FRONT_AGENT_SYSTEM),
        ("SQL_GENERATION_SYSTEM template",                   SQL_GENERATION_SYSTEM),
        ("SQL_GENERATION_RETRY_HINT (only on retries)",      SQL_GENERATION_RETRY_HINT),
        ("CHART_SPEC_SYSTEM",                                CHART_SPEC_SYSTEM),
        ("CHART_FORCE_NOTE (only on rechart/force)",         CHART_FORCE_NOTE),
        ("SUMMARIZE_SYSTEM template",                        SUMMARIZE_SYSTEM),
    ]

    print(f"{'Tokens':>7}  {'Chars':>7}  Component")
    print(f"{'-'*7}  {'-'*7}  ---------")
    for label, text in components:
        n = n_tokens(text)
        print(f"{n:>7,}  {len(text):>7,}  {label}")

    # --- Assembled (what actually goes over the wire each call) ---
    print()
    print("Per-call cost projection (input tokens only, no prompt caching):")
    print()

    assembled = [
        ("front_agent system",
         FRONT_AGENT_SYSTEM.format(dataset_notes=DATASET_NOTES)),
        ("generate_sql system (no retry)",
         SQL_GENERATION_SYSTEM.format(dataset_notes=DATASET_NOTES, schema=schema)),
        ("generate_sql system (retry: +hint with prior SQL & error)",
         SQL_GENERATION_SYSTEM.format(dataset_notes=DATASET_NOTES, schema=schema)
         + SQL_GENERATION_RETRY_HINT.format(prior_sql="-- 200 char SQL --", prior_error="-- 200 char error --")),
        ("summarize system",
         SUMMARIZE_SYSTEM.format(dataset_notes=DATASET_NOTES)),
        ("chart picker system",
         CHART_SPEC_SYSTEM),
        ("chart picker system (force mode)",
         CHART_FORCE_NOTE + CHART_SPEC_SYSTEM),
    ]

    print(f"{'Tokens':>7}  {'$ / 1k calls':>13}  Assembly")
    print(f"{'-'*7}  {'-'*13}  --------")
    for label, text in assembled:
        n = n_tokens(text)
        print(f"{n:>7,}  {cost_per_1k_calls(n):>13}  {label}")

    # --- Per-turn aggregate (rough) ---
    print()
    print("Rough per-turn input token totals (system prompts only — excludes user msg, history, tool results):")
    print()

    front_assembled  = FRONT_AGENT_SYSTEM.format(dataset_notes=DATASET_NOTES)
    sql_assembled    = SQL_GENERATION_SYSTEM.format(dataset_notes=DATASET_NOTES, schema=schema)
    sumr_assembled   = SUMMARIZE_SYSTEM.format(dataset_notes=DATASET_NOTES)
    chart_assembled  = CHART_SPEC_SYSTEM

    n_front = n_tokens(front_assembled)
    n_sql   = n_tokens(sql_assembled)
    n_sumr  = n_tokens(sumr_assembled)
    n_chart = n_tokens(chart_assembled)

    data_path = n_front + n_sql + n_sumr + n_chart
    respond_path = n_front
    rechart_path = n_front + n_chart  # chart picker bypassed when override is set, but worst case

    print(f"  data path    (front + sql + summarize + chart): {data_path:>7,} tokens, "
          f"{cost_per_1k_calls(data_path)} per 1k turns")
    print(f"  respond path (front only):                       {respond_path:>7,} tokens, "
          f"{cost_per_1k_calls(respond_path)} per 1k turns")
    print(f"  rechart path (front + chart):                    {rechart_path:>7,} tokens, "
          f"{cost_per_1k_calls(rechart_path)} per 1k turns")

    print()
    print("Note: assumes no prompt caching. Wiring Anthropic's prompt-caching")
    print("      header on the static schema would cut input cost ~10x on repeat calls.")


if __name__ == "__main__":
    main()
