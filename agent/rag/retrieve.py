"""Runtime retrieval against the rag_embeddings corpus.

Given a question, embed it (Voyage with input_type='query' — different from
the 'document' embeddings used by the indexer; both are required for good
asymmetric retrieval) and pull the top-K most-similar payloads from the
glossary or example corpora.

The result is formatted as a markdown block that gets injected into the
SQL generator's USER message. Keeping retrieval in the user message
(not the system prefix) preserves the system-prompt cache hit — system
stays stable across requests, only the user-facing block varies.

Failures degrade gracefully: empty list returned, the SQL generator
falls back to its cached prompt without retrieval. A RAG-down state
shouldn't break the agent.
"""
from __future__ import annotations

import json
import logging
import os
from functools import cache
from typing import Any

from sqlalchemy import text

from agent.db import agent_engine

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "voyage-3"
DEFAULT_K_GLOSSARY = 4
DEFAULT_K_EXAMPLES = 3
# Cosine similarity floor — entries below this are presumed off-topic and
# filtered out. Surfacing low-confidence retrievals as "relevant" misleads
# the SQL generator more than missing context does. Empirically tuned
# for voyage-3: relevant matches typically score 0.40-0.65, marginal
# matches 0.30-0.40, off-topic <0.30. Threshold 0.35 keeps the
# marginal-but-relevant tier in scope.
SIMILARITY_THRESHOLD = 0.35


@cache
def _voyage_client():
    # Imported lazily so the agent doesn't blow up at boot if voyageai
    # isn't installed in some deployment context — retrieval just degrades.
    import voyageai
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    return voyageai.Client(api_key=api_key)


def _to_pgvector(emb: list[float]) -> str:
    """pgvector accepts the bracketed string form natively. Faster than the
    array literal and avoids SQLAlchemy type-binding gymnastics."""
    return "[" + ",".join(f"{x:.7f}" for x in emb) + "]"


def _retrieve(question: str, kind: str, k: int) -> list[dict]:
    """Embed + similarity search. Returns list of {id, payload, similarity}.

    Empty list on any failure (Voyage down, DB unreachable, no rows).
    Caller treats empty as "no retrieved context" — silent degradation.
    """
    try:
        result = _voyage_client().embed(
            [question],
            model=EMBEDDING_MODEL,
            input_type="query",  # asymmetric: docs were embedded as 'document'
        )
        emb = result.embeddings[0]
    except Exception as e:
        logger.warning("RAG embed failed (%s) — degrading to no retrieval", e)
        return []

    emb_str = _to_pgvector(emb)
    try:
        with agent_engine().connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT id, payload,
                           1 - (embedding <=> CAST(:emb AS vector)) AS similarity
                    FROM rag_embeddings
                    WHERE kind = :kind
                    ORDER BY embedding <=> CAST(:emb AS vector)
                    LIMIT :k
                """),
                {"emb": emb_str, "kind": kind, "k": k},
            ).all()
    except Exception as e:
        logger.warning("RAG query failed (%s) — degrading to no retrieval", e)
        return []

    out = []
    for r in rows:
        sim = float(r.similarity)
        if sim < SIMILARITY_THRESHOLD:
            continue
        # payload is JSONB → already a dict from psycopg's json adapter
        payload = r.payload if isinstance(r.payload, dict) else json.loads(r.payload)
        out.append({"id": r.id, "payload": payload, "similarity": sim})
    return out


def retrieve_glossary(question: str, k: int = DEFAULT_K_GLOSSARY) -> list[dict]:
    return _retrieve(question, "glossary", k)


def retrieve_examples(question: str, k: int = DEFAULT_K_EXAMPLES) -> list[dict]:
    return _retrieve(question, "example", k)


# ───────────────────────── prompt formatting ─────────────────────────────

def _format_glossary_entry(hit: dict) -> str:
    p = hit["payload"]
    sim = hit["similarity"]
    parts = [f"- **{p['term']}** (sim {sim:.2f}): {p.get('definition', '').strip()}"]
    conventions = (p.get("conventions") or "").strip()
    if conventions:
        parts.append("  Conventions:")
        for line in conventions.splitlines():
            parts.append(f"  {line}")
    return "\n".join(parts)


def _format_example_entry(hit: dict, n: int) -> str:
    p = hit["payload"]
    sim = hit["similarity"]
    parts = [f"**Example {n}** — {p['question']!r} (sim {sim:.2f}):"]
    parts.append("```sql")
    parts.append((p.get("sql") or "").strip())
    parts.append("```")
    notes = (p.get("notes") or "").strip()
    if notes:
        parts.append(f"_Notes: {notes}_")
    return "\n".join(parts)


def format_context_block(glossary: list[dict], examples: list[dict]) -> str:
    """Render retrieved hits as a markdown block for the SQL user prompt.
    Empty string when nothing was retrieved — caller can detect and skip
    rendering the surrounding 'Retrieved context:' header."""
    if not glossary and not examples:
        return ""
    sections = []
    if glossary:
        sections.append("**Relevant business glossary:**\n")
        sections.extend(_format_glossary_entry(h) for h in glossary)
    if examples:
        sections.append("\n**Relevant example queries:**\n")
        for i, h in enumerate(examples, 1):
            sections.append(_format_example_entry(h, i))
            sections.append("")  # blank line between examples
    return "\n".join(sections).rstrip()


def retrieve_context_block(
    question: str,
    *,
    k_glossary: int = DEFAULT_K_GLOSSARY,
    k_examples: int = DEFAULT_K_EXAMPLES,
) -> tuple[str, list[dict], list[dict]]:
    """One-call helper: embed once, fetch both corpora, format. Returns
    (formatted_block, glossary_hits, example_hits) so callers can also
    surface counts/IDs for observability."""
    glossary = retrieve_glossary(question, k_glossary)
    examples = retrieve_examples(question, k_examples)
    return format_context_block(glossary, examples), glossary, examples


if __name__ == "__main__":
    # Tiny CLI for spot-checking retrievals. Usage:
    #   python -m agent.rag.retrieve "show me revenue by country"
    import sys

    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("usage: python -m agent.rag.retrieve <question>")
        raise SystemExit(2)
    q = " ".join(sys.argv[1:])
    block, g, e = retrieve_context_block(q)
    print(f"Question: {q!r}\n")
    print(f"Glossary hits ({len(g)}):")
    for h in g:
        print(f"  · {h['id']}  sim={h['similarity']:.3f}")
    print(f"\nExample hits ({len(e)}):")
    for h in e:
        print(f"  · {h['id']}  sim={h['similarity']:.3f}")
    print("\n" + ("─" * 60) + "\n")
    print(block or "(no context retrieved)")
