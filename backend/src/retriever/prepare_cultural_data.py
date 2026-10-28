"""Prepare local cultural datasets for retrieval and mediator profiling.

The utility is deliberately dependency-light for JSON/JSONL/CSV inputs. Parquet
support is optional and uses pandas/pyarrow when available; it processes rows in
small batches so large shards in ``data/datasets`` are safe to use.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

from src.config import BASE_DIR

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_INPUT_DIR = BASE_DIR / "data" / ("dataset" if (BASE_DIR / "data" / "dataset").exists() else "datasets")
KNOWLEDGE_PATH = PROCESSED_DIR / "cultural_knowledge_chunks.jsonl"
MEDIATOR_PATH = PROCESSED_DIR / "mediator_training_data.jsonl"
SUPPORTED = {".json", ".jsonl", ".csv", ".parquet"}

SLANG_MARKERS: dict[str, tuple[str, ...]] = {
    "Tamil (Tamil Nadu / Chennai Slang)": (
        "machi", "macha", "mapla", "naina", "thambi", "akka", "anna", "pa", "da", "di",
        "vanakkam", "vaanga", "ponga", "sollunga", "eppadi", "irukinga", "romba", "nalla",
        "konjam", "ippo", "enna", "yaaru", "saaptiya", "nanba", "saptiya", "la",
    ),
    "Hindi (Hinglish)": (
        "arey", "arre", "bhai", "behen", "aap", "aapka", "aapki", "aapko", "kaise", "kaisa",
        "kaisi", "hai", "hain", "haan", "kya", "kyun", "kyon", "main", "mujhe", "mera",
        "meri", "mere", "tum", "tumhara", "tumhari", "tujhe", "tera", "teri", "nahi", "nahin",
        "bahut", "bohot", "achha", "accha", "theek", "thik", "yaar", "dost", "batao", "bolo",
        "chalo", "phir", "abhi", "ghar", "wala", "wali", "namaste", "shukriya", "ji",
    ),
}


def normalize_text(value: Any) -> str:
    """Normalize Unicode, whitespace, and common dataset formatting noise."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_markers(text: str) -> tuple[str, list[str]]:
    """Return the strongest lexical culture label and matched markers."""
    lowered = normalize_text(text).lower()
    scores = {label: [token for token in tokens if re.search(rf"\b{re.escape(token)}\b", lowered)] for label, tokens in SLANG_MARKERS.items()}
    label, markers = max(scores.items(), key=lambda item: len(item[1]))
    return (label, markers) if markers else ("Neutral / Unresolved", [])


def chunk_words(text: str, size: int = 512, overlap: int = 64) -> Iterator[str]:
    words = text.split()
    if not words:
        return
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if chunk:
            yield chunk
        if start + size >= len(words):
            break


def _record_text(record: dict[str, Any]) -> str:
    preferred = ("text", "content", "question", "query", "answer", "context", "sentence")
    values = [record.get(key) for key in preferred if record.get(key)]
    if not values:
        values = [value for value in record.values() if isinstance(value, (str, int, float))]
    return normalize_text(" ".join(map(str, values)))


def iter_records(path: Path, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    """Yield records from supported local formats without requiring one schema."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    yield value if isinstance(value, dict) else {"text": value}
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)
        for value in data if isinstance(data, list) else [data]:
            yield value if isinstance(value, dict) else {"text": value}
        return
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet preprocessing requires pyarrow.") from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
        return
    return


def prepare_dataset(input_dir: str | Path, output_dir: str | Path = PROCESSED_DIR, limit: int | None = None) -> tuple[int, int]:
    """Create cultural retrieval chunks and mediator examples from local files."""
    input_path, output_path = Path(input_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    knowledge_count = mediator_count = seen = 0
    with (output_path / KNOWLEDGE_PATH.name).open("w", encoding="utf-8") as knowledge, (output_path / MEDIATOR_PATH.name).open("w", encoding="utf-8") as mediator:
        for path in sorted(input_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            try:
                records = iter_records(path)
                for record in records:
                    if limit is not None and seen >= limit:
                        return knowledge_count, mediator_count
                    if "tradition_name" in record:
                        t_name = record["tradition_name"]
                        t_reg = record.get("region", "Tamil Nadu")
                        t_reality = record.get("little_tradition_reality", "")
                        t_guide = record.get("expected_explanation_guideline", "")
                        t_domain = record.get("domain", "Everyday Commonsense")
                        steps = "\n".join(record.get("customary_process_steps", []))
                        advice = record.get("actionable_decision_advice", "")
                        chunk = (
                            f"Tradition: {t_name}. Region: {t_reg}. "
                            f"Cultural Context & Norm: {t_reality} "
                            f"Customary Process Steps: {steps} "
                            f"Decision & Practical Guidance: {advice}"
                        )
                        knowledge.write(
                            json.dumps(
                                {
                                    "region": t_reg,
                                    "dialect_tag": record.get("dialect_name", "Tamil"),
                                    "domain": t_domain,
                                    "cultural_practice_fact": chunk,
                                    "source": path.name,
                                    "chunk_index": 0,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        knowledge_count += 1
                        for q in record.get("sample_tanglish_queries", []):
                            mediator.write(
                                json.dumps(
                                    {
                                        "user_query": q,
                                        "slang_indicators": record.get("tanglish_slang_triggers", []),
                                        "dialect_label": record.get("dialect_name", "Tamil"),
                                        "target_mediated_output": f"Explain the tradition of {t_name}: {t_reality}",
                                        "source": path.name,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            mediator_count += 1
                        seen += 1
                        continue

                    text = _record_text(record)
                    if len(text) < 20:
                        continue
                    culture, markers = detect_markers(text)
                    for index, chunk in enumerate(chunk_words(text)):
                        knowledge.write(json.dumps({"region": "Tamil Nadu / South Asia" if culture.startswith("Tamil") else "South Asia", "dialect_tag": culture, "domain": "Everyday Commonsense", "cultural_practice_fact": chunk, "source": path.name, "chunk_index": index}, ensure_ascii=False) + "\n")
                        knowledge_count += 1
                    mediator.write(json.dumps({"user_query": text[:1000], "slang_indicators": markers, "dialect_label": culture, "target_mediated_output": f"Use the {culture} context and explain any regional variation before answering.", "source": path.name}, ensure_ascii=False) + "\n")
                    mediator_count += 1
                    seen += 1
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
    return knowledge_count, mediator_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local cultural datasets for CAL mediation.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    knowledge, mediator = prepare_dataset(args.input_dir, args.output_dir, args.limit)
    print(f"Prepared {knowledge} knowledge chunks and {mediator} mediator examples.")


if __name__ == "__main__":
    main()
