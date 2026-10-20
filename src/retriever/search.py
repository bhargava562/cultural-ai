"""Cultural context retriever and semantic similarity searcher."""

import argparse
import sys
from typing import Any, Optional
from rich.console import Console

# Force UTF-8 on Windows consoles to render Tamil scripts properly
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.retriever.ingest import get_chroma_client, get_or_create_collection

console = Console(highlight=False)


class CulturalRetriever:
    """Local Chroma retriever with optional culture/domain metadata filters."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self.client = get_chroma_client(persist_dir)
        self.collection = get_or_create_collection(self.client)

    def search(self, query: str, profile: Any = None, top_k: int = 3, domain: Optional[str] = None) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        culture = getattr(profile, "detected_culture", None)
        if culture and not culture.startswith("Neutral"):
            where["dialect_tag"] = culture
        if domain:
            where["domain"] = domain
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(top_k, self.collection.count())}
        if where:
            kwargs["where"] = {"$and": [{key: value} for key, value in where.items()]}
        if not self.collection.count():
            return []
        try:
            results = self.collection.query(**kwargs)
        except Exception:
            # A sparse local corpus should still return semantically useful facts.
            kwargs.pop("where", None)
            results = self.collection.query(**kwargs)
        return _results_to_snippets(results)


def _results_to_snippets(results: dict[str, Any]) -> list[dict[str, Any]]:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else []
    return [{"content": doc, "metadata": metadatas[i] if i < len(metadatas) else {}, "distance": distances[i] if i < len(distances) else None} for i, doc in enumerate(documents)]


def query_cultural_context(
    query: str,
    n_results: int = 3,
    persist_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k culturally relevant context snippets from ChromaDB."""
    return CulturalRetriever(persist_dir).search(query, top_k=n_results)


def format_cultural_context(snippets: list[dict[str, Any]]) -> str:
    """Format retrieved snippets into a clear string block for LLM prompts."""
    if not snippets:
        return "No specific local cultural archive snippets retrieved."

    formatted_lines: list[str] = ["### Retrieved Cultural & Linguistic Context:"]
    for i, s in enumerate(snippets, start=1):
        title = s.get("metadata", {}).get("title", f"Cultural Snippet {i}")
        formatted_lines.append(f"[{i}] {title}:\n{s['content']}\n")
    return "\n".join(formatted_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local cultural knowledge index.")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    snippets = query_cultural_context(args.query, n_results=args.top_k)
    console.print(f"[bold green]Retrieved {len(snippets)} snippets for query:[/bold green] '{args.query}'\n")
    console.print(format_cultural_context(snippets))


if __name__ == "__main__":
    main()
