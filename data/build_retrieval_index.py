"""Build retrieval indices (BM25 / ColBERT) from Wikipedia corpus.

BM25: Uses rank_bm25 for in-memory indexing (suitable for <1M passages).
ColBERT: Uses ColBERTv2 for dense retrieval (optional, requires GPU).
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def build_bm25_index(corpus_path: Path, index_path: Path, k1: float = 0.9, b: float = 0.4):
    """Build and save a BM25 index from corpus JSONL.

    Args:
        corpus_path: Path to corpus JSONL file.
        index_path: Path to save pickled BM25 index.
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
    """
    from rank_bm25 import BM25Okapi

    logger.info("Loading corpus from %s", corpus_path)
    documents = []
    tokenized_corpus = []

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
            # Simple whitespace tokenization
            text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            tokenized_corpus.append(text.split())

    logger.info("Building BM25 index for %d documents (k1=%.1f, b=%.1f)", len(documents), k1, b)
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = {
        "bm25": bm25,
        "documents": documents,
        "params": {"k1": k1, "b": b},
    }
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info("BM25 index saved to %s (%d docs)", index_path, len(documents))


def build_colbert_index(
    corpus_path: Path,
    index_path: Path,
    checkpoint: str = "colbert-ir/colbertv2.0",
    nbits: int = 2,
):
    """Build a ColBERT v2 index from corpus JSONL.

    Requires: pip install colbert-ai[faiss-gpu]

    Args:
        corpus_path: Path to corpus JSONL file.
        index_path: Path to save ColBERT index.
        checkpoint: ColBERT checkpoint name.
        nbits: Quantization bits for residual compression.
    """
    try:
        from colbert import Indexer
        from colbert.infra import ColBERTConfig, Run, RunConfig
    except ImportError:
        logger.error("ColBERT not installed. Install with: pip install colbert-ai[faiss-gpu]")
        return

    # Prepare collection file (ColBERT expects TSV: id \t text)
    collection_path = index_path.parent / "collection.tsv"
    collection_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing collection for ColBERT indexing...")
    with open(corpus_path, encoding="utf-8") as fin, \
         open(collection_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            doc = json.loads(line.strip())
            text = f"{doc.get('title', '')} {doc.get('text', '')}".replace("\t", " ").replace("\n", " ")
            fout.write(f"{i}\t{text}\n")

    logger.info("Building ColBERT index (checkpoint=%s, nbits=%d)", checkpoint, nbits)
    with Run().context(RunConfig(nranks=1)):
        config = ColBERTConfig(
            nbits=nbits,
            doc_maxlen=300,
            index_path=str(index_path),
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(
            name=index_path.name,
            collection=str(collection_path),
        )

    logger.info("ColBERT index saved to %s", index_path)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus JSONL path")
    parser.add_argument("--engine", type=str, default="bm25", choices=["bm25", "colbert"])
    parser.add_argument("--index_path", type=str, required=True, help="Output index path")
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b")
    parser.add_argument("--checkpoint", type=str, default="colbert-ir/colbertv2.0")
    args = parser.parse_args()

    if args.engine == "bm25":
        build_bm25_index(Path(args.corpus), Path(args.index_path), k1=args.k1, b=args.b)
    elif args.engine == "colbert":
        build_colbert_index(Path(args.corpus), Path(args.index_path), checkpoint=args.checkpoint)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
