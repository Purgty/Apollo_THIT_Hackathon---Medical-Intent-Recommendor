"""
Apollo Clinical Pipeline — Stage 1: Semantic Text Creation
===========================================================
This script ingests the raw Apollo Product Catalog (CSV) and produces a
derived dataset where each product row carries a ``product_semantic_text``
field — a dense, human-readable text document synthesising the product's
clinical identity.

Design rationale
----------------
Classical keyword search (BM25 / TF-IDF) fails when users express clinical
needs in natural language (e.g., "medicine for my baby's runny nose") rather
than exact product names. By constructing a semantic document per product and
later embedding it with a Sentence Transformer, we enable approximate-nearest-
neighbour retrieval that generalises across medical synonyms and paraphrases.

The ``product_semantic_text`` is constructed by concatenating the following
catalog fields in priority order:
  1. Product Name       — Brand / generic identity
  2. Key Benefits/Uses  — Indication signal (highest retrieval value)
  3. Product Information — Detailed clinical context
  4. Classifiers 1-3   — ATC / therapeutic category hierarchy
  5. Product Type       — Broad category signal
  6. Molecules          — Active pharmaceutical ingredient (API) signal
  7. Pharma Composition — Detailed formulation context

Run:
    conda activate nlp
    python final_implementation/data_pipeline/semantic_text_creation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Make project root importable regardless of invocation directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from config.settings import (
    CATALOG_METADATA_COLUMNS,
    CATALOG_RAW_PATH,
    CATALOG_TEXT_COLUMNS,
    SEMANTIC_CATALOG_PATH,
)
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

def clean_field(value: object) -> str:
    """
    Normalise a single catalog field value.

    - NaN / None → empty string (never propagate null)
    - Collapse internal whitespace
    - Strip leading / trailing whitespace
    - Lowercase for consistency across lookup and query
    """
    if pd.isna(value):
        return ""
    text = str(value)
    # Collapse unicode whitespace and control characters
    text = " ".join(text.split())
    return text.strip().lower()


def build_semantic_document(row: pd.Series) -> str:
    """
    Assemble a single rich semantic text document for one product row.

    Each field is labelled with a semantic prefix to preserve contextual
    distinctiveness during the embedding phase. This approach mirrors
    techniques from biomedical NLP literature (e.g., PubMedBERT fine-tuning)
    where structured prompts improve retrieval precision.

    Returns
    -------
    str
        A space-joined document string.  Empty fields are silently skipped.
    """
    sections = [
        ("product", row.get("name", "")),
        ("benefits", row.get("Key Benefits/Uses", "")),
        ("description", row.get("Product Information", "")),
        ("category", row.get("Classifier 1", "")),
        ("subcategory", row.get("Classifier 2", "")),
        ("subsubcategory", row.get("Classifier 3", "")),
        ("type", row.get("Product Type 1", "")),
        ("molecules", row.get("molecules", "")),
        ("composition", row.get("Pharma Composition", "")),
    ]
    parts: list[str] = []
    for _label, value in sections:
        cleaned = clean_field(value)
        if cleaned:
            parts.append(cleaned)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    console.rule("[bold blue]Apollo Clinical Pipeline — Stage 1: Semantic Text Creation")
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load raw catalog
    # ------------------------------------------------------------------
    console.print(f"\n[cyan]Loading catalog from:[/cyan] {CATALOG_RAW_PATH}")
    if not CATALOG_RAW_PATH.exists():
        log.error("catalog_not_found", extra={"path": str(CATALOG_RAW_PATH)})
        console.print(f"[red]ERROR:[/red] Catalog file not found at {CATALOG_RAW_PATH}")
        sys.exit(1)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Reading CSV (may take a moment for large file)...", total=None)
        df = pd.read_csv(
            CATALOG_RAW_PATH,
            encoding="latin1",
            low_memory=False,
        )
        progress.update(task, completed=True)

    log.info("catalog_loaded", extra={"rows": len(df), "columns": list(df.columns)})
    console.print(f"  Loaded [bold]{len(df):,}[/bold] products with [bold]{len(df.columns)}[/bold] columns.\n")

    # ------------------------------------------------------------------
    # 2. Validate expected columns are present
    # ------------------------------------------------------------------
    missing = [c for c in CATALOG_TEXT_COLUMNS if c not in df.columns]
    if missing:
        log.warning("columns_missing_from_catalog", extra={"missing": missing})
        console.print(f"[yellow]Warning:[/yellow] These expected columns are absent and will be skipped: {missing}")

    # ------------------------------------------------------------------
    # 3. Clean text columns in-place
    # ------------------------------------------------------------------
    console.print("[cyan]Cleaning text fields...[/cyan]")
    for col in CATALOG_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_field)

    # ------------------------------------------------------------------
    # 4. Build semantic documents
    # ------------------------------------------------------------------
    console.print("[cyan]Building semantic documents...[/cyan]")
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task(f"Generating semantic text for {len(df):,} products...", total=None)
        df["product_semantic_text"] = df.apply(build_semantic_document, axis=1)
        progress.update(task, completed=True)

    # ------------------------------------------------------------------
    # 5. Preview sample output
    # ------------------------------------------------------------------
    table = Table(title="Semantic Text Preview (first 3 products)", show_lines=True)
    table.add_column("Product Name", style="green", max_width=40)
    table.add_column("Semantic Text Snippet", max_width=80)
    for _, row in df.head(3).iterrows():
        table.add_row(row.get("name", "N/A"), row["product_semantic_text"][:200])
    console.print(table)

    # ------------------------------------------------------------------
    # 6. Select metadata columns + semantic text and save
    # ------------------------------------------------------------------
    available_meta_cols = [c for c in CATALOG_METADATA_COLUMNS if c in df.columns]
    df_output = df[available_meta_cols + ["product_semantic_text"]].copy()

    # Drop rows where semantic text is empty (data quality guard)
    empty_count = (df_output["product_semantic_text"].str.strip() == "").sum()
    if empty_count:
        log.warning("empty_semantic_text_rows_dropped", extra={"count": int(empty_count)})
        console.print(f"[yellow]Warning:[/yellow] Dropping {empty_count:,} rows with empty semantic text.")
        df_output = df_output[df_output["product_semantic_text"].str.strip() != ""]

    df_output.to_csv(SEMANTIC_CATALOG_PATH, index=False)
    elapsed = time.perf_counter() - start
    log.info(
        "semantic_catalog_saved",
        extra={
            "output_path": str(SEMANTIC_CATALOG_PATH),
            "rows": len(df_output),
            "elapsed_seconds": round(elapsed, 2),
        },
    )
    console.print(
        f"\n[bold green]✓ Stage 1 complete![/bold green] "
        f"Saved [bold]{len(df_output):,}[/bold] rows to:\n  {SEMANTIC_CATALOG_PATH}\n"
        f"  Elapsed: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    run()
