"""Phase 1: Generate search-reasoning trajectories for all 3 datasets.

Usage:
    python -m scripts.run_phase1_generation \
        --data_dir /root/autodl-tmp \
        --model /root/autodl-tmp/models/Qwen3-32B-AWQ \
        --max_questions 500
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def gpu_precheck(threshold_mb: int = 1000) -> None:
    """Check GPU memory and attempt cleanup if occupied."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        mem_used = int(result.stdout.strip().split("\n")[0])
        if mem_used > threshold_mb:
            logging.warning("GPU has %dMB in use. Attempting cleanup (pkill vllm)...", mem_used)
            subprocess.run(["pkill", "-f", "vllm"], timeout=10)
            time.sleep(3)
            # Re-check
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            mem_after = int(result.stdout.strip().split("\n")[0])
            if mem_after > threshold_mb:
                logging.error("GPU still has %dMB after cleanup. Aborting.", mem_after)
                sys.exit(1)
            logging.info("GPU cleanup successful: %dMB → %dMB", mem_used, mem_after)
        else:
            logging.info("GPU memory OK: %dMB used", mem_used)
    except Exception as e:
        logging.warning("GPU precheck failed (non-fatal): %s", e)
logger = logging.getLogger(__name__)

DATASETS = [
    {"name": "hotpotqa", "file": "hotpotqa_train.jsonl"},
    {"name": "musique", "file": "musique_train.jsonl"},
    {"name": "2wikimhqa", "file": "2wikimhqa_train.jsonl"},
]


def load_questions(path: Path, max_questions: int) -> list[dict]:
    """Load questions from JSONL, return list of {id, question, answer}."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            questions.append({
                "id": obj["id"],
                "question": obj["question"],
                "answer": obj.get("answer", ""),
            })
            if len(questions) >= max_questions:
                break
    return questions


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Trajectory generation")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/models/Qwen3-32B-AWQ")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--num_traj", type=int, default=3)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset(s) to run, comma-separated (e.g. 'hotpotqa' or 'musique,2wikimhqa'). Default: all")
    parser.add_argument("--suffix", type=str, default="",
                        help="Output file suffix (e.g. 'v2' → hotpotqa_trajectories_v2.jsonl)")
    args = parser.parse_args()

    gpu_precheck()

    data_dir = Path(args.data_dir)
    corpus_path = data_dir / "corpus" / "wikipedia_subset.jsonl"
    datasets_dir = data_dir / "datasets"
    output_dir = data_dir / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        sys.exit(1)

    # Filter datasets if specified (supports comma-separated names)
    datasets_to_run = DATASETS
    if args.dataset:
        requested = [s.strip() for s in args.dataset.split(",")]
        datasets_to_run = [d for d in DATASETS if d["name"] in requested]
        if not datasets_to_run:
            logger.error("Unknown dataset: %s", args.dataset)
            sys.exit(1)

    # Initialize search engine (once, shared across datasets)
    logger.info("Loading BM25 search engine from %s", corpus_path)
    from generation.search_engine import BM25SearchEngine
    engine = BM25SearchEngine(corpus_path=str(corpus_path))

    # Initialize trajectory generator (once, shared across datasets)
    logger.info("Initializing trajectory generator with %s (tp=%d)", args.model, args.tp)
    from generation.generate_trajectories import TrajectoryGenerator
    generator = TrajectoryGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tp,
    )

    # Generate for each dataset
    summary = {}
    for ds in datasets_to_run:
        ds_name = ds["name"]
        input_path = datasets_dir / ds["file"]

        if not input_path.exists():
            logger.warning("Dataset file not found: %s, skipping", input_path)
            summary[ds_name] = {"status": "skipped", "reason": "file not found"}
            continue

        suffix = f"_{args.suffix}" if args.suffix else ""
        output_path = output_dir / f"{ds_name}_trajectories{suffix}.jsonl"
        logger.info("=== Dataset: %s ===", ds_name)

        questions = load_questions(input_path, args.max_questions)
        logger.info("Loaded %d questions from %s", len(questions), input_path)

        start = time.time()
        trajectories = generator.generate_batch(
            questions=questions,
            search_engine=engine,
            num_trajectories_per_question=args.num_traj,
            output_path=str(output_path),
        )
        elapsed = time.time() - start

        # Count valid trajectories (has answer and at least 1 search step)
        valid = [t for t in trajectories if not t["incomplete"] and t["num_search_steps"] > 0]
        has_answer = [t for t in valid if "Answer:" in t["text"]]

        summary[ds_name] = {
            "status": "done",
            "total": len(trajectories),
            "valid": len(valid),
            "has_answer": len(has_answer),
            "incomplete": len([t for t in trajectories if t["incomplete"]]),
            "elapsed_s": round(elapsed, 1),
            "output": str(output_path),
        }
        logger.info(
            "Dataset %s: %d total, %d valid, %d with answer, %d incomplete (%.1fs)",
            ds_name, len(trajectories), len(valid), len(has_answer),
            len(trajectories) - len(valid), elapsed,
        )

    # Final summary
    logger.info("=== PHASE 1 SUMMARY ===")
    total_valid = sum(s.get("valid", 0) for s in summary.values())
    for ds_name, stats in summary.items():
        logger.info("  %s: %s", ds_name, json.dumps(stats))
    logger.info("Total valid trajectories: %d (gate: >=500)", total_valid)

    if total_valid < 500:
        logger.warning("GATE FAILED: Only %d valid trajectories (<500). Strategy adjustment needed.", total_valid)
        sys.exit(2)
    else:
        logger.info("GATE PASSED: %d valid trajectories (>=500)", total_valid)

    # Write summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
