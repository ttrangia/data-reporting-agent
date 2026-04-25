"""Live check that prompt caching is actually firing.

Hits the SQL-generation prompt twice with the same input. First call should
report `cache_creation_input_tokens` (cache write); second call should report
`cache_read_input_tokens` ~= the cached size at ~10% of normal input cost.

Run with:
    venv/bin/python -m tools.verify_cache
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic

from agent.db import pagila_schema_string
from agent.prompts import (
    DATASET_NOTES,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
)


def cached_system(text: str) -> dict:
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
        ],
    }


def show(label: str, msg) -> None:
    """Pretty-print the input/cache split from an AIMessage's usage metadata."""
    usage = getattr(msg, "usage_metadata", None) or {}
    details = usage.get("input_token_details", {}) or {}
    cache_read = details.get("cache_read", 0)
    cache_create = details.get("cache_creation", 0)
    base_input = usage.get("input_tokens", 0)
    output = usage.get("output_tokens", 0)
    print(f"\n{label}")
    print(f"  input_tokens (uncached portion) : {base_input:>6,}")
    print(f"  cache_creation_input_tokens     : {cache_create:>6,}  (write — pays ~125% rate)")
    print(f"  cache_read_input_tokens         : {cache_read:>6,}  (read  — pays  ~10% rate)")
    print(f"  output_tokens                   : {output:>6,}")


def main() -> None:
    schema = pagila_schema_string()
    system = SQL_GENERATION_SYSTEM.format(dataset_notes=DATASET_NOTES, schema=schema)
    user = SQL_GENERATION_USER.format(retry_context="", question="How many active customers do we have?")

    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

    # Call 1 — expect cache write
    msg1 = llm.invoke([cached_system(system), {"role": "user", "content": user}])
    show("CALL 1 (expect cache_creation > 0):", msg1)

    # Call 2 — same prompt, expect cache read
    msg2 = llm.invoke([cached_system(system), {"role": "user", "content": user}])
    show("CALL 2 (expect cache_read > 0):", msg2)

    # Note: 5-minute TTL on the cache, refreshed each hit. Subsequent runs of
    # this script within 5 minutes also hit the cache for both calls.


if __name__ == "__main__":
    main()
