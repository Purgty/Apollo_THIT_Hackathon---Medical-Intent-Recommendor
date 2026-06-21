"""
Apollo Clinical Pipeline — Stage 2: Vector Embeddings & FAISS Index Builder
=============================================================================
This script reads the semantic catalog produced by Stage 1, generates dense
vector embeddings using a Sentence Transformer model, and persists a FAISS
similarity index and metadata lookup table.

Architecture notes
------------------
Embedding model: ``all-MiniLM-L6-v2``
    A 22M-parameter bi-encoder fine-tuned on 1 billion+ training pairs.
    Produces 384-dimensional L2-normalised embeddings. Selected for its
    optimal balance of semantic richness and inference speed on CPU hardware.

FAISS index strategy:
    - ``flat`` mode  →  IndexFlatL2: brute-force L2 search.  Exact results.
      Recommended for ≤ 500k vectors (our catalog = 85k).  Sub-millisecond
      query time after index load.
    - ``ivf``  mode  →  IndexIVFFlat: IVF partitioning with ``nlist`` Voronoi
      cells and ``nprobe`` probes at query time.  Scales to tens of millions
      of vectors with a controllable accuracy / speed tradeoff.

The index is persisted to disk alongside a Pickle of the metadata DataFrame
so that the lookup engine can load both in < 2 seconds at API startup.

Run:
    conda activate nlp
    python final_implementation/data_pipeline/vector_index_builder.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "final_implementation"))

from config.settings import (
    ARTIFACTS_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_INDEX_TYPE,
    FAISS_IVF_NLIST,
    FAISS_IVF_NPROBE,
    METADATA_PKL_PATH,
    SEMANTIC_CATALOG_PATH,
)
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> SentenceTransformer:
    console.print(f"[cyan]Loading embedding model:[/cyan] {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    elapsed = time.perf_counter() - t0
    log.info("embedding_model_loaded", extra={"model": model_name, "elapsed_s": round(elapsed, 2)})
    console.print(f"  Model ready in {elapsed:.1f}s\n")
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode a list of strings to L2-normalised float32 embeddings.

    L2-normalisation is applied so that inner-product search (IndexFlatIP)
    becomes equivalent to cosine similarity, giving more intuitive distance
    semantics.  IndexFlatL2 on normalised vectors is equivalent to cosine
    distance ranking.
    """
    console.print(f"[cyan]Encoding {len(texts):,} texts (batch size {batch_size})...[/cyan]")
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Embedding products...", total=len(texts))
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(batch_emb)
            progress.advance(task, len(batch))

    embeddings = np.vstack(embeddings_list).astype("float32")

    # L2-normalise each row vector
    faiss.normalize_L2(embeddings)

    elapsed = time.perf_counter() - t0
    log.info(
        "embeddings_generated",
        extra={
            "shape": list(embeddings.shape),
            "elapsed_s": round(elapsed, 2),
            "throughput_per_s": round(len(texts) / elapsed, 1),
        },
    )
    console.print(f"  Embeddings shape: {embeddings.shape} | Elapsed: {elapsed:.1f}s\n")
    return embeddings


# ---------------------------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = FAISS_INDEX_TYPE,
    nlist: int = FAISS_IVF_NLIST,
    nprobe: int = FAISS_IVF_NPROBE,
) -> faiss.Index:
    """
    Build and return a FAISS index populated with all embeddings.

    Parameters
    ----------
    embeddings:  float32 numpy array of shape (N, D)
    index_type:  "flat" or "ivf"
    nlist:       Number of Voronoi cells for IVF (ignored for flat)
    nprobe:      Number of cells probed at query time for IVF
    """
    n, d = embeddings.shape
    console.print(f"[cyan]Building FAISS index ({index_type}) for {n:,} × {d}-dim vectors...[/cyan]")
    t0 = time.perf_counter()

    if index_type == "ivf":
        # IVF: faster for large N at the cost of approximate results
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        log.info("faiss_ivf_training", extra={"nlist": nlist})
        index.train(embeddings)  # type: ignore[attr-defined]
        index.nprobe = nprobe    # type: ignore[attr-defined]
    else:
        # Flat: exact, optimal for our 85k-vector catalog
        index = faiss.IndexFlatL2(d)

    index.add(embeddings)
    elapsed = time.perf_counter() - t0
    log.info(
        "faiss_index_built",
        extra={
            "index_type": index_type,
            "total_vectors": index.ntotal,
            "elapsed_s": round(elapsed, 2),
        },
    )
    console.print(f"  Index type: {index_type.upper()} | Vectors: {index.ntotal:,} | Elapsed: {elapsed:.1f}s\n")
    return index


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    console.rule("[bold blue]Apollo Clinical Pipeline — Stage 2: Vector Index Builder")
    overall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load semantic catalog
    # ------------------------------------------------------------------
    console.print(f"\n[cyan]Loading semantic catalog from:[/cyan] {SEMANTIC_CATALOG_PATH}")
    if not SEMANTIC_CATALOG_PATH.exists():
        log.error("semantic_catalog_not_found", extra={"path": str(SEMANTIC_CATALOG_PATH)})
        console.print(
            "[red]ERROR:[/red] Semantic catalog not found. Run Stage 1 first:\n"
            "  python final_implementation/data_pipeline/semantic_text_creation.py"
        )
        sys.exit(1)

    df = pd.read_csv(SEMANTIC_CATALOG_PATH, low_memory=False)
    assert "product_semantic_text" in df.columns, \
        "Column 'product_semantic_text' missing — re-run Stage 1."

    console.print(f"  Loaded [bold]{len(df):,}[/bold] products.\n")
    log.info("semantic_catalog_loaded", extra={"rows": len(df)})

    # ------------------------------------------------------------------
    # 2. Load embedding model and encode
    # ------------------------------------------------------------------
    model = load_model(EMBEDDING_MODEL_NAME)
    texts: list[str] = df["product_semantic_text"].fillna("").tolist()
    embeddings = encode_texts(model, texts)

    assert embeddings.shape == (len(df), EMBEDDING_DIMENSION), (
        f"Expected shape ({len(df)}, {EMBEDDING_DIMENSION}), got {embeddings.shape}"
    )

    # ------------------------------------------------------------------
    # 3. Build FAISS index
    # ------------------------------------------------------------------
    index = build_faiss_index(embeddings)

    # ------------------------------------------------------------------
    # 4. Persist index and metadata
    # ------------------------------------------------------------------
    console.print(f"[cyan]Saving FAISS index to:[/cyan] {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    console.print(f"[cyan]Saving metadata pickle to:[/cyan] {METADATA_PKL_PATH}")
    # Reset index to guarantee alignment between FAISS row IDs and DataFrame.iloc
    df_meta = df.reset_index(drop=True)
    with open(METADATA_PKL_PATH, "wb") as fh:
        pickle.dump(df_meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

    overall_elapsed = time.perf_counter() - overall_start
    log.info(
        "stage2_complete",
        extra={
            "faiss_index_path": str(FAISS_INDEX_PATH),
            "metadata_pkl_path": str(METADATA_PKL_PATH),
            "total_elapsed_s": round(overall_elapsed, 2),
        },
    )
    console.print(
        f"\n[bold green]✓ Stage 2 complete![/bold green] "
        f"FAISS index ({index.ntotal:,} vectors) and metadata saved.\n"
        f"  Elapsed: {overall_elapsed:.1f}s"
    )


if __name__ == "__main__":
    run()
