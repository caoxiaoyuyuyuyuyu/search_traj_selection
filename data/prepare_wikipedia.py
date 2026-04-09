"""Prepare Wikipedia subset for local retrieval.

Extracts Wikipedia passages relevant to multi-hop QA datasets (HotpotQA, MuSiQue,
2WikiMHQA) and builds a unified corpus for BM25/ColBERT indexing.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_titles_from_dataset(dataset_path: Path) -> set[str]:
    """Extract all Wikipedia titles referenced in a dataset file.

    Looks at supporting_facts and context passages to identify needed articles.
    """
    titles: set[str] = set()
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            # From supporting facts
            for sf in record.get("supporting_facts", []):
                if isinstance(sf, dict) and "title" in sf:
                    titles.add(sf["title"])
                elif isinstance(sf, list) and len(sf) >= 1:
                    titles.add(sf[0])
    return titles


def load_hotpotqa_context_passages(hf_dataset_split) -> dict[str, list[str]]:
    """Extract context passages from HotpotQA's built-in context field.

    HotpotQA distractor setting includes context paragraphs alongside each question.
    """
    passages: dict[str, list[str]] = defaultdict(list)
    for example in hf_dataset_split:
        if "context" in example:
            ctx = example["context"]
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                text = " ".join(sentences)
                if text and text not in passages[title]:
                    passages[title].append(text)
    return dict(passages)


def build_corpus(
    dataset_files: list[Path],
    output_path: Path,
    include_hotpotqa_context: bool = True,
) -> int:
    """Build unified Wikipedia corpus from dataset references.

    Args:
        dataset_files: Processed dataset JSONL files.
        output_path: Output JSONL path for corpus.
        include_hotpotqa_context: If True, also extract HotpotQA's built-in passages.

    Returns:
        Number of passages written.
    """
    # Collect all referenced titles
    all_titles: set[str] = set()
    for path in dataset_files:
        titles = extract_titles_from_dataset(path)
        all_titles.update(titles)
        logger.info("  %s: %d unique titles", path.name, len(titles))

    logger.info("Total unique titles across datasets: %d", len(all_titles))

    # For a full implementation, we would:
    # 1. Download Wikipedia dump (e.g., from HuggingFace's wikipedia dataset)
    # 2. Filter to only needed articles
    # 3. Split into passages
    # Here we provide the pipeline; actual Wikipedia data loading depends on source

    # Extract passages from HotpotQA context (which includes Wikipedia paragraphs)
    passages_by_title: dict[str, list[str]] = defaultdict(list)

    if include_hotpotqa_context:
        try:
            from datasets import load_dataset
            logger.info("Loading HotpotQA for context passages...")
            ds = load_dataset("hotpot_qa", "distractor", split="train")
            hqa_passages = load_hotpotqa_context_passages(ds)
            for title, texts in hqa_passages.items():
                passages_by_title[title].extend(texts)
            logger.info("Extracted passages for %d titles from HotpotQA context", len(hqa_passages))
        except Exception as e:
            logger.warning("Could not load HotpotQA context: %s", e)

    # Write corpus
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_passages = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for title, texts in sorted(passages_by_title.items()):
            for i, text in enumerate(texts):
                doc = {
                    "id": f"{title}_{i}",
                    "title": title,
                    "text": text,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_passages += 1

    logger.info("Wrote %d passages to %s", n_passages, output_path)
    return n_passages


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia corpus for retrieval")
    parser.add_argument("--dataset_dir", type=str, default="data/processed",
                        help="Directory with processed dataset JSONL files")
    parser.add_argument("--output", type=str, default="data/corpus/wikipedia_subset.jsonl",
                        help="Output corpus JSONL path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_files = list(dataset_dir.glob("*_train.jsonl"))
    if not dataset_files:
        logger.error("No training dataset files found in %s", dataset_dir)
        return

    logger.info("Building corpus from %d dataset files", len(dataset_files))
    build_corpus(dataset_files, Path(args.output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
