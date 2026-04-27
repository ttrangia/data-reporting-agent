"""Offline indexer for the RAG corpus.

Reads `agent/rag/glossary.yaml` and `agent/rag/examples.yaml`, computes
the embedding-text projection for each entry, hashes it, and upserts
into the `rag_embeddings` Postgres table. Only re-embeds entries whose
hash changed — re-running is cheap.

Usage:
    python -m agent.rag.build_index                # incremental (default)
    python -m agent.rag.build_index --force        # re-embed everything
    python -m agent.rag.build_index --dry-run      # show plan, don't write
    python -m agent.rag.build_index --kind glossary  # only one corpus

Environment:
    DATABASE_URL_ADMIN — read-write Postgres role (NOT report_agent).
    VOYAGE_API_KEY     — embedding API key. Get one at voyageai.com/dashboard.

The schema must already be applied — run `psql "$DATABASE_URL_ADMIN" -f
agent/rag/schema.sql` once before the first indexer run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

# Imported here so the import error is loud and obvious if the package
# isn't installed (pip install voyageai pgvector).
import voyageai  # noqa: E402

from agent.db import admin_engine  # noqa: E402

# voyage-3 returns 1024-dim embeddings. Switching models means re-running
# with --force AND updating the vector(1024) dim in schema.sql.
EMBEDDING_MODEL = "voyage-3"
EMBEDDING_DIM = 1024

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOSSARY_PATH = REPO_ROOT / "agent" / "rag" / "glossary.yaml"
EXAMPLES_PATH = REPO_ROOT / "agent" / "rag" / "examples.yaml"


@dataclass
class IndexEntry:
    """One row to upsert. `embed_text` is what gets embedded; `payload` is
    what the SQL generator sees at retrieval time."""
    id: str
    kind: str            # 'glossary' | 'example'
    embed_text: str
    payload: dict
    content_hash: str


# ───────────────────────── projections ──────────────────────────────────

def _glossary_embed_text(entry: dict) -> str:
    """Term + aliases + first paragraph of definition. Aliases broaden
    the match radius; the first paragraph is the densest semantic
    signal — later paragraphs are conventions/edge cases that come
    along in the payload but shouldn't dilute the embedding."""
    parts = [entry["term"]]
    aliases = entry.get("aliases") or []
    if aliases:
        parts.append(" / ".join(aliases))
    definition = entry.get("definition", "").strip()
    first_para = definition.split("\n\n")[0] if definition else ""
    if first_para:
        parts.append(first_para)
    return " | ".join(parts)


def _example_embed_text(entry: dict) -> str:
    """Question + alias_questions. SQL is NEVER embedded — it shares
    tokens with every other query and adds noise to the cosine space."""
    parts = [entry["question"]]
    aliases = entry.get("alias_questions") or []
    parts.extend(aliases)
    return " | ".join(parts)


PROJECTIONS = {
    "glossary": _glossary_embed_text,
    "example":  _example_embed_text,
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_corpus(kind: str, path: Path) -> Iterable[IndexEntry]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a YAML list at top level.")
    for entry in raw:
        if "id" not in entry:
            raise ValueError(f"Entry without id in {path}: {entry!r}")
        embed_text = PROJECTIONS[kind](entry)
        yield IndexEntry(
            id=f"{kind}:{entry['id']}",
            kind=kind,
            embed_text=embed_text,
            payload=entry,
            content_hash=_hash(embed_text),
        )


# ───────────────────────── DB helpers ───────────────────────────────────

def _existing_hashes() -> dict[str, str]:
    """Pull current (id, content_hash) pairs so we can skip unchanged rows."""
    with admin_engine().connect() as conn:
        rows = conn.execute(text("SELECT id, content_hash FROM rag_embeddings")).all()
    return {r.id: r.content_hash for r in rows}


def _upsert(entries: list[IndexEntry], embeddings: list[list[float]]) -> None:
    """One transaction, parameterized — pgvector accepts a Python list and
    SQLAlchemy/psycopg handles the cast to vector via the explicit ::vector."""
    assert len(entries) == len(embeddings)
    with admin_engine().begin() as conn:
        for e, emb in zip(entries, embeddings):
            conn.execute(
                text("""
                    INSERT INTO rag_embeddings (id, kind, content_hash, embedding, payload, embedded_at)
                    VALUES (:id, :kind, :hash, CAST(:emb AS vector), CAST(:payload AS jsonb), now())
                    ON CONFLICT (id) DO UPDATE SET
                        kind         = EXCLUDED.kind,
                        content_hash = EXCLUDED.content_hash,
                        embedding    = EXCLUDED.embedding,
                        payload      = EXCLUDED.payload,
                        embedded_at  = now()
                """),
                {
                    "id":      e.id,
                    "kind":    e.kind,
                    "hash":    e.content_hash,
                    # pgvector accepts the bracketed string form natively
                    "emb":     "[" + ",".join(f"{x:.7f}" for x in emb) + "]",
                    "payload": json.dumps(e.payload, default=str),
                },
            )


def _delete_orphans(active_ids: set[str]) -> int:
    """Drop rows whose source YAML entry was deleted."""
    with admin_engine().begin() as conn:
        existing = {r.id for r in conn.execute(text("SELECT id FROM rag_embeddings")).all()}
        orphans = existing - active_ids
        if not orphans:
            return 0
        conn.execute(
            text("DELETE FROM rag_embeddings WHERE id = ANY(:ids)"),
            {"ids": list(orphans)},
        )
        return len(orphans)


# ───────────────────────── main ─────────────────────────────────────────

def _embed_batch(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    """Single batched call. Voyage's batch limit is well above any corpus
    size we'll have here — comfortable to send everything in one shot."""
    if not texts:
        return []
    result = client.embed(texts, model=EMBEDDING_MODEL, input_type="document")
    return result.embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/refresh the RAG embedding index.")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every entry, ignoring the hash skip.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without calling the embedding API or writing.")
    parser.add_argument("--kind", choices=["glossary", "example", "all"], default="all",
                        help="Restrict to one corpus.")
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = []
    if args.kind in ("glossary", "all"):
        sources.append(("glossary", GLOSSARY_PATH))
    if args.kind in ("example", "all"):
        sources.append(("example",  EXAMPLES_PATH))

    all_entries: list[IndexEntry] = []
    for kind, path in sources:
        loaded = list(_load_corpus(kind, path))
        all_entries.extend(loaded)
        print(f"  {kind}: {len(loaded)} entries in {path.name}")

    # Dry-run is fully offline — verify YAML + projections without ever
    # touching Postgres or the embedding API.
    if args.dry_run:
        print(f"\n[dry-run] would consider {len(all_entries)} entries")
        for e in all_entries:
            print(f"  {e.id}  ({len(e.embed_text)} chars, hash={e.content_hash[:10]}…)")
        return 0

    if not os.getenv("VOYAGE_API_KEY"):
        print("error: VOYAGE_API_KEY not set in environment.", file=sys.stderr)
        return 2
    if not os.getenv("DATABASE_URL_ADMIN"):
        print("error: DATABASE_URL_ADMIN not set in environment.", file=sys.stderr)
        return 2

    existing = {} if args.force else _existing_hashes()
    to_embed = [e for e in all_entries if existing.get(e.id) != e.content_hash]
    skipped = len(all_entries) - len(to_embed)

    print(f"\nPlan: {len(to_embed)} to embed, {skipped} unchanged (skip).")

    if to_embed:
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        print(f"Embedding {len(to_embed)} entries via {EMBEDDING_MODEL}…")
        embeddings = _embed_batch(client, [e.embed_text for e in to_embed])
        if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
            print(f"error: model returned dim={len(embeddings[0])}, expected {EMBEDDING_DIM}.",
                  file=sys.stderr)
            return 3
        print(f"Upserting {len(embeddings)} rows…")
        _upsert(to_embed, embeddings)

    if args.kind == "all":
        active_ids = {e.id for e in all_entries}
        orphan_count = _delete_orphans(active_ids)
        if orphan_count:
            print(f"Deleted {orphan_count} orphan row(s) (entry removed from YAML).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
