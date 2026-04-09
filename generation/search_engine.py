"""Local search engine wrapper for trajectory generation.

Provides a unified interface for BM25 and ColBERT retrieval over a Wikipedia corpus.
Used during teacher trajectory generation to simulate web search.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A retrieved document."""
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self, max_chars: int = 1000) -> str:
        """Format as context string for LLM consumption."""
        text = self.text[:max_chars]
        return f"[{self.title}] {text}"


class SearchEngine(ABC):
    """Abstract search engine interface."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Search the corpus and return top-k documents.

        Args:
            query: Search query string.
            top_k: Number of documents to return.

        Returns:
            List of Document objects sorted by relevance (descending).
        """
        ...

    @abstractmethod
    def batch_search(self, queries: list[str], top_k: int = 5) -> list[list[Document]]:
        """Batch search for efficiency.

        Args:
            queries: List of query strings.
            top_k: Number of documents per query.

        Returns:
            List of document lists, one per query.
        """
        ...


class BM25SearchEngine(SearchEngine):
    """BM25 retrieval engine using rank_bm25.

    Loads a Wikipedia corpus from JSONL and builds an in-memory BM25 index.
    For larger corpora, consider pyserini with a pre-built Lucene index.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        k1: float = 0.9,
        b: float = 0.4,
        index_path: str | Path | None = None,
    ):
        self.corpus_path = Path(corpus_path)
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, str]] = []
        self.bm25 = None

        self._load_corpus()
        self._build_index()

    def _load_corpus(self):
        """Load Wikipedia corpus from JSONL file."""
        logger.info("Loading corpus from %s", self.corpus_path)
        self.documents = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                self.documents.append(doc)
        logger.info("Loaded %d documents", len(self.documents))

    def _build_index(self):
        """Build BM25 index from loaded documents."""
        from rank_bm25 import BM25Okapi

        logger.info("Building BM25 index (k1=%.1f, b=%.1f)", self.k1, self.b)
        tokenized_corpus = [
            self._tokenize(doc.get("text", "") + " " + doc.get("title", ""))
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercasing tokenizer."""
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(Document(
                doc_id=doc.get("id", str(idx)),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                score=float(scores[idx]),
            ))
        return results

    def batch_search(self, queries: list[str], top_k: int = 5) -> list[list[Document]]:
        return [self.search(q, top_k) for q in queries]


class ColBERTSearchEngine(SearchEngine):
    """ColBERT v2 retrieval engine.

    Uses the ColBERT library for dense retrieval. Requires a pre-built index.
    This is optional and will be integrated later when dense retrieval is needed.
    """

    def __init__(
        self,
        index_path: str | Path,
        checkpoint: str = "colbert-ir/colbertv2.0",
        top_k: int = 5,
    ):
        self.index_path = Path(index_path)
        self.checkpoint = checkpoint
        self.default_top_k = top_k
        self.searcher = None

        self._load_index()

    def _load_index(self):
        """Load pre-built ColBERT index."""
        try:
            from colbert import Searcher
            from colbert.infra import ColBERTConfig

            logger.info("Loading ColBERT index from %s", self.index_path)
            config = ColBERTConfig(index_root=str(self.index_path.parent))
            self.searcher = Searcher(
                index=self.index_path.name,
                config=config,
                checkpoint=self.checkpoint,
            )
            logger.info("ColBERT index loaded")
        except ImportError:
            logger.warning(
                "ColBERT not installed. Install with: pip install colbert-ai[faiss-gpu]"
            )
            raise

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        if self.searcher is None:
            raise RuntimeError("ColBERT index not loaded")

        results = self.searcher.search(query, k=top_k)
        documents = []
        for passage_id, rank, score in zip(*results):
            doc_text = self.searcher.collection[passage_id]
            documents.append(Document(
                doc_id=str(passage_id),
                title="",  # ColBERT passages may not have separate titles
                text=doc_text,
                score=float(score),
            ))
        return documents

    def batch_search(self, queries: list[str], top_k: int = 5) -> list[list[Document]]:
        if self.searcher is None:
            raise RuntimeError("ColBERT index not loaded")

        # ColBERT supports efficient batch search
        all_results = self.searcher.search_all({i: q for i, q in enumerate(queries)}, k=top_k)
        batch_docs = []
        for i in range(len(queries)):
            docs = []
            for passage_id, rank, score in all_results.data.get(i, []):
                doc_text = self.searcher.collection[passage_id]
                docs.append(Document(
                    doc_id=str(passage_id),
                    title="",
                    text=doc_text,
                    score=float(score),
                ))
            batch_docs.append(docs)
        return batch_docs


def create_search_engine(
    engine_type: str = "bm25",
    **kwargs: Any,
) -> SearchEngine:
    """Factory function to create a search engine.

    Args:
        engine_type: "bm25" or "colbert".
        **kwargs: Engine-specific parameters.

    Returns:
        SearchEngine instance.
    """
    if engine_type == "bm25":
        return BM25SearchEngine(**kwargs)
    elif engine_type == "colbert":
        return ColBERTSearchEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Choose 'bm25' or 'colbert'.")
