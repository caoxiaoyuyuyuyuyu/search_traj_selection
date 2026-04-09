"""Download and unify multi-hop QA datasets for search trajectory selection.

Supports: HotpotQA (distractor), MuSiQue, 2WikiMultiHopQA.
Output: unified JSONL with fields {id, question, answer, supporting_facts, dataset, split}.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-dataset extractors
# ---------------------------------------------------------------------------

def extract_hotpotqa(example: dict[str, Any]) -> dict[str, Any]:
    """Extract from HotpotQA distractor setting."""
    supporting = []
    if "supporting_facts" in example:
        sf = example["supporting_facts"]
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        supporting = [
            {"title": t, "sent_id": s} for t, s in zip(titles, sent_ids)
        ]
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "supporting_facts": supporting,
    }


def extract_musique(example: dict[str, Any]) -> dict[str, Any]:
    """Extract from MuSiQue dataset."""
    supporting = []
    if "paragraphs" in example:
        for p in example["paragraphs"]:
            if p.get("is_supporting", False):
                supporting.append({
                    "title": p.get("title", ""),
                    "paragraph_text": p.get("paragraph_text", ""),
                })
    return {
        "id": example.get("id", ""),
        "question": example["question"],
        "answer": example.get("answer", ""),
        "supporting_facts": supporting,
    }


def extract_2wikimhqa(example: dict[str, Any]) -> dict[str, Any]:
    """Extract from 2WikiMultiHopQA."""
    supporting = []
    if "supporting_facts" in example:
        sf = example["supporting_facts"]
        if isinstance(sf, dict):
            titles = sf.get("title", [])
            sent_ids = sf.get("sent_id", [])
            supporting = [
                {"title": t, "sent_id": s} for t, s in zip(titles, sent_ids)
            ]
        elif isinstance(sf, list):
            supporting = [{"title": s[0], "sent_id": s[1]} for s in sf if len(s) >= 2]
    return {
        "id": example.get("_id", example.get("id", "")),
        "question": example["question"],
        "answer": example.get("answer", ""),
        "supporting_facts": supporting,
    }


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "hotpotqa": {
        "hf_path": "hotpot_qa",
        "hf_config": "distractor",
        "splits": {"train": "train", "dev": "validation"},
        "extractor": extract_hotpotqa,
    },
    "musique": {
        "hf_path": "drt/musique",
        "hf_config": None,
        "splits": {"train": "train", "dev": "validation"},
        "extractor": extract_musique,
    },
    "2wikimhqa": {
        "hf_path": "scholarly-shadows/2WikiMultiHopQA",
        "hf_config": None,
        "splits": {"train": "train", "dev": "validation"},
        "extractor": extract_2wikimhqa,
    },
}


def download_and_convert(
    dataset_name: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    """Download one dataset and write unified JSONL files.

    Returns:
        Dict mapping split name to number of examples written.
    """
    cfg = DATASETS[dataset_name]
    logger.info("Downloading %s (path=%s, config=%s)", dataset_name, cfg["hf_path"], cfg["hf_config"])

    load_kwargs: dict[str, Any] = {"path": cfg["hf_path"]}
    if cfg["hf_config"]:
        load_kwargs["name"] = cfg["hf_config"]
    ds = load_dataset(**load_kwargs, trust_remote_code=True)

    counts: dict[str, int] = {}
    for split_name, hf_split in cfg["splits"].items():
        if hf_split not in ds:
            logger.warning("Split %s not found in %s, skipping", hf_split, dataset_name)
            continue

        split_data = ds[hf_split]
        out_path = output_dir / f"{dataset_name}_{split_name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for example in split_data:
                record = cfg["extractor"](example)
                record["dataset"] = dataset_name
                record["split"] = split_name
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                if max_samples and n >= max_samples:
                    break

        counts[split_name] = n
        logger.info("  %s/%s: %d examples → %s", dataset_name, split_name, n, out_path)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Download multi-hop QA datasets")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for JSONL files")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()),
                        help="Which datasets to download")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split (for debugging)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary: dict[str, dict[str, int]] = {}

    for name in args.datasets:
        counts = download_and_convert(name, output_dir, args.max_samples)
        summary[name] = counts

    logger.info("=== Download Summary ===")
    for name, counts in summary.items():
        total = sum(counts.values())
        logger.info("  %s: %d total (%s)", name, total, ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
