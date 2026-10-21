"""Ingestion pipeline for Cultural RAG into local persistent ChromaDB."""

import argparse
import sys
import logging
import os
from pathlib import Path
from typing import Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console

# Force UTF-8 on Windows consoles to render Tamil scripts properly
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import settings, BASE_DIR

console = Console(highlight=False)
logger = logging.getLogger("cultural_ai.ingest")

COLLECTION_NAME = "tamil_cultural_knowledge"

# Curated seed cultural context for zero-cost immediate bootstrapping
SAMPLE_CULTURAL_DOCS = [
    {
        "id": "doc_pongal_traditions",
        "title": "Thai Pongal Celebrations & Etiquette",
        "content": (
            "தைப்பொங்கல் (Thai Pongal) is the prime harvest festival of Tamils celebrated over 4 days: "
            "Bhogi (போகி), Surya Pongal (சூரியப் பொங்கல்), Mattu Pongal (மாட்டுப் பொங்கல்), and Kaanum Pongal (காணும் பொங்கல்). "
            "Tradition requires boiling milk and new rice in an earthen pot decorated with ginger and turmeric leaves. "
            "When milk boils over, family members shout 'Pongalo Pongal!' (பொங்கலோ பொங்கல்). "
            "Inviting elders requires deep respect (Maryadhai): traditionally done in person or with polite formal language "
            "using honorific plural verbs (vaanga, ponga, vanthu sirappikka vendum) rather than informal pronouns."
        ),
        "source": "curated_taxonomy",
        "category": "festivals",
    },
    {
        "id": "doc_honorifics_kinship",
        "title": "Tamil Honorific Address & Kinship System",
        "content": (
            "Tamil culture places immense emphasis on age-appropriate and relationship-specific address. "
            "Never use singular 'nee' (நீ) for anyone older or unfamiliar; use 'neenga' (நீங்கள்) and verbs ending in '-unga' "
            "(e.g., vaanga, ukkarunga, sollunga). "
            "Kinship titles: Mother's elder sister is 'Periyamma' (பெரியம்மா), younger sister is 'Chithi' (சித்தி). "
            "Father's elder brother is 'Periyappa' (பெரியப்பா), younger brother is 'Chithappa' (சித்தப்பா). "
            "Maternal uncle is 'Mama' (மாமா), paternal aunt is 'Athai' (அத்தை). "
            "Greeting an elder traditionally includes folded hands (Vanakkam) and inquiring after their wellbeing (Nalamaa / Sowkyamaa)."
        ),
        "source": "curated_taxonomy",
        "category": "kinship",
    },
    {
        "id": "doc_virundhombal_hospitality",
        "title": "Virundhombal: Tamil Hospitality Norms",
        "content": (
            "விருந்தோம்பல் (Virundhombal) is an essential pillar described in Thirukkural. "
            "When guests enter a Tamil home, offering a tumbler of water (Thanneer) is immediate and mandatory before business or talk. "
            "Guests must always be invited to share a meal (Saapittu ponga). "
            "Refusing food completely can cause polite embarrassment; accepting at least tea, coffee, or prasadam is customary. "
            "When seeing guests off, one never says 'Poittu vaa' (go); instead say 'Poyittu varugiren' / 'Poyittu vaanga' (go and return)."
        ),
        "source": "curated_taxonomy",
        "category": "social_norms",
    },
    {
        "id": "doc_tanglish_chennai_slang",
        "title": "Tanglish Pragmatics and Chennai Dialect Nuances",
        "content": (
            "Tanglish blends English syntax with Tamil pragmatic markers (da, di, pa, machi, la). "
            "In Chennai colloquial speech: 'Machi' / 'Mappillai' denotes deep fraternal friendship between peers. "
            "'Naina' means elder/father figure in North Chennai slang. "
            "'Galatta' means playful chaos or noisy celebration. "
            "Markers like 'pa' soften commands into friendly requests ('Konjam kelunga pa'). "
            "Casual markers like 'da' or 'di' should NEVER be directed at elders or professional acquaintances."
        ),
        "source": "curated_taxonomy",
        "category": "linguistics",
    },
    {
        "id": "doc_kongu_tamil_warmth",
        "title": "Kongu Tamil Politeness & Idioms",
        "content": (
            "Kongu Tamil (Coimbatore, Erode, Tiruppur) is renowned for gentle, continuous politeness markers. "
            "Verbal suffix '-unga' is added even when addressing younger relatives or children out of affectionate respect "
            "(e.g., 'enna kannu solreenga', 'saapteengalaa'). "
            "Hospitality is marked by persistent offers of home-cooked food and warm familial inquiries."
        ),
        "source": "curated_taxonomy",
        "category": "linguistics",
    },
]


def get_chroma_client(persist_dir: Optional[str] = None) -> chromadb.PersistentClient:
    """Initialize persistent local ChromaDB client."""
    path = persist_dir or settings.chroma_persist_dir
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the cultural knowledge collection with sentence transformers or default ONNX embedding."""
    # Do not download a 400+ MB transformer while importing the API. Opt into
    # the local Sentence-Transformers model for ingestion/retrieval explicitly.
    # Chroma's default function is lightweight to construct and defers its own
    # model work until a query is actually made.
    if os.getenv("CAL_USE_SENTENCE_TRANSFORMERS", "false").lower() in {"1", "true", "yes"}:
        try:
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
        except Exception as exc:
            logger.info("SentenceTransformerEmbeddingFunction unavailable (%s); using Chroma default embeddings.", exc)
            emb_fn = embedding_functions.DefaultEmbeddingFunction()
    else:
        emb_fn = embedding_functions.DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"description": "Tamil and regional socio-cultural context chunks"},
    )


def ingest_sample_docs(collection: chromadb.Collection) -> int:
    """Ingest curated sample documents and regional traditions into ChromaDB."""
    ids = [doc["id"] for doc in SAMPLE_CULTURAL_DOCS]
    documents = [doc["content"] for doc in SAMPLE_CULTURAL_DOCS]
    metadatas = [
        {"title": doc["title"], "source": doc["source"], "category": doc["category"], "dialect_tag": "Tamil"}
        for doc in SAMPLE_CULTURAL_DOCS
    ]

    # Ingest authoritative regional traditions if present
    traditions_file = BASE_DIR / "data" / "datasets" / "tn_regional_traditions_tanglish.json"
    if traditions_file.exists():
        try:
            import json
            with open(traditions_file, "r", encoding="utf-8") as f:
                traditions = json.load(f)
            for t in traditions:
                t_id = t["id"]
                content = (
                    f"Tradition: {t['tradition_name']}.\n"
                    f"Region: {t['region']}.\n"
                    f"Cultural Context: {t['little_tradition_reality']}\n"
                    f"Guideline: {t['expected_explanation_guideline']}\n"
                    f"Misconception cleared: {t.get('great_tradition_misconception', '')}\n"
                    f"Tanglish triggers: {', '.join(t.get('tanglish_slang_triggers', []))}"
                )
                ids.append(t_id)
                documents.append(content)
                metadatas.append({
                    "title": t["tradition_name"],
                    "source": "tn_regional_traditions_tanglish.json",
                    "category": t.get("domain", "traditions"),
                    "dialect_tag": t.get("dialect_name", "Tamil"),
                    "region": t.get("region", "Tamil Nadu"),
                })
        except Exception as exc:
            logger.warning("Failed to load traditions from %s: %s", traditions_file, exc)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    console.print(
        f"[green]Successfully ingested {len(ids)} curated cultural & regional tradition documents into ChromaDB collection '{COLLECTION_NAME}'.[/green]"
    )
    return len(ids)


def ingest_huggingface_dataset(
    collection: chromadb.Collection,
    dataset_name: str,
    text_column: str = "text",
    limit: int = 500,
    batch_size: int = 50,
) -> int:
    """Ingest documents from a Hugging Face dataset in streaming/batched mode."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print(
            "[red]The 'datasets' package is required for Hugging Face ingestion. Please install it using 'pip install datasets'.[/red]"
        )
        return 0

    console.print(
        f"[cyan]Streaming dataset '{dataset_name}' (limit: {limit}, batch: {batch_size})...[/cyan]"
    )
    ds = load_dataset(dataset_name, split="train", streaming=True)

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    total_ingested = 0

    for idx, item in enumerate(ds):
        if total_ingested >= limit:
            break

        text = item.get(text_column) or item.get("content") or item.get("sentence")
        if not text or len(str(text).strip()) < 30:
            continue

        doc_id = f"hf_{dataset_name.replace('/', '_')}_{idx}"
        batch_ids.append(doc_id)
        batch_docs.append(str(text)[:1500])
        batch_meta.append({"source": dataset_name, "index": idx})

        if len(batch_ids) >= batch_size:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
            total_ingested += len(batch_ids)
            console.print(f"Ingested {total_ingested}/{limit} records...")
            batch_ids.clear()
            batch_docs.clear()
            batch_meta.clear()

    if batch_ids:
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        total_ingested += len(batch_ids)

    console.print(f"[green]Completed Hugging Face ingestion: {total_ingested} docs.[/green]")
    return total_ingested


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split text into chunks of roughly chunk_size characters/words with chunk_overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_local_dataset_file(
    collection: chromadb.Collection,
    file_path: str | Path,
    text_column: str = "text",
    limit: Optional[int] = None,
    batch_size: int = 50,
) -> int:
    """Ingest a local dataset file (.parquet, .json, .jsonl, .csv)."""
    import json
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return 0

    console.print(f"[cyan]Ingesting local dataset file: {path.name}...[/cyan]")
    texts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    ext = path.suffix.lower()
    if ext == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            col = text_column if text_column in df.columns else df.columns[0]
            texts = df[col].dropna().astype(str).tolist()
            metadata_rows = [{"source": path.name, "row": i} for i in range(len(texts))]
        except ImportError:
            console.print("[yellow]pandas / pyarrow required for parquet ingestion. Please install them or upload jsonl/csv.[/yellow]")
            return 0
    elif ext == ".csv":
        import csv
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            col = text_column if reader.fieldnames and text_column in reader.fieldnames else (reader.fieldnames[0] if reader.fieldnames else None)
            if col:
                for row in reader:
                    if row.get(col):
                        texts.append(row[col])
                        metadata_rows.append({"source": path.name, "row": len(texts) - 1})
    elif ext in [".jsonl", ".json"]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        val = obj.get(text_column) or obj.get("content") or obj.get("sentence") or (list(obj.values())[0] if obj else "")
                        if val:
                            texts.append(str(val))
                            metadata_rows.append({"source": path.name, "row": len(texts) - 1})
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            val = item.get(text_column) or item.get("content") or item.get("sentence")
                            if val:
                                texts.append(str(val))
                                metadata_rows.append({"source": path.name, "row": len(texts) - 1})
                        elif isinstance(item, str):
                            texts.append(item)
                            metadata_rows.append({"source": path.name, "row": len(texts) - 1})

    if not texts:
        console.print(f"[yellow]No texts extracted from {path.name}[/yellow]")
        return 0

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    total_ingested = 0

    for idx, raw_text in enumerate(texts):
        if limit and total_ingested >= limit:
            break

        chunks = chunk_text(raw_text, chunk_size=512, chunk_overlap=64)
        source_meta = metadata_rows[idx] if idx < len(metadata_rows) else {"source": path.name, "row": idx}
        for c_idx, chunk in enumerate(chunks):
            doc_id = f"local_{path.stem}_{idx}_{c_idx}"
            batch_ids.append(doc_id)
            batch_docs.append(chunk)
            batch_meta.append({**source_meta, "chunk": c_idx})

            if len(batch_ids) >= batch_size:
                collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
                total_ingested += len(batch_ids)
                batch_ids.clear()
                batch_docs.clear()
                batch_meta.clear()

    if batch_ids:
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        total_ingested += len(batch_ids)

    console.print(f"[green]Successfully ingested {total_ingested} chunks from {path.name}.[/green]")
    return total_ingested


def ingest_local_directory(
    collection: chromadb.Collection,
    dir_path: str | Path,
    text_column: str = "text",
    limit: Optional[int] = None,
) -> int:
    """Ingest all supported dataset files from a local directory."""
    path = Path(dir_path)
    if not path.exists():
        console.print(f"[red]Directory not found: {path}[/red]")
        return 0

    total = 0
    supported = [".parquet", ".json", ".jsonl", ".csv"]
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in supported:
            total += ingest_local_dataset_file(collection, f, text_column=text_column, limit=limit)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest cultural data into local ChromaDB.")
    parser.add_argument("--sample", action="store_true", help="Ingest curated sample documents")
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name")
    parser.add_argument("--file", type=str, default=None, help="Path to local dataset file (.parquet, .jsonl, .csv)")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory containing dataset files (e.g., data/datasets)")
    parser.add_argument("--text-col", type=str, default="text", help="Text column in dataset")
    parser.add_argument("--limit", type=int, default=500, help="Max items to process")
    args = parser.parse_args()

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if args.file:
        ingest_local_dataset_file(collection, args.file, text_column=args.text_col, limit=args.limit)

    if args.dir:
        ingest_local_directory(collection, args.dir, text_column=args.text_col, limit=args.limit)

    if args.dataset:
        ingest_huggingface_dataset(
            collection,
            dataset_name=args.dataset,
            text_column=args.text_col,
            limit=args.limit,
        )

    if args.sample or (not args.dataset and not args.file and not args.dir and not args.sample):
        ingest_sample_docs(collection)


if __name__ == "__main__":
    main()
