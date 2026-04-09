"""Multi-hop QA evaluation: Exact Match (EM) and token-level F1.

Evaluates trained student models on HotpotQA, MuSiQue, and 2WikiMHQA dev sets.
"""

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (standard HotpotQA evaluation)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract the final answer from model output.

    Looks for patterns like:
    - "The answer is: ..."
    - "Final answer: ..."
    - Last sentence as fallback
    """
    # Try explicit answer patterns
    patterns = [
        r"(?:the answer is|final answer)[:\s]+(.+?)(?:\.|$)",
        r"(?:answer)[:\s]+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate a trained student model on multi-hop QA."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        logger.info("Loading model from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate_answer(self, question: str) -> str:
        """Generate an answer for a question."""
        messages = [
            {"role": "user", "content": f"Answer the following question concisely.\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # greedy for evaluation
            temperature=1.0,
        )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate_dataset(
        self,
        dataset_path: str | Path,
        max_samples: int | None = None,
    ) -> dict:
        """Evaluate on a dataset file.

        Args:
            dataset_path: Path to JSONL with "question" and "answer" fields.
            max_samples: Max number of examples to evaluate.

        Returns:
            Dict with em, f1 (averages), and per-example results.
        """
        examples = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        if max_samples:
            examples = examples[:max_samples]

        em_scores = []
        f1_scores = []
        results = []

        for i, ex in enumerate(examples):
            if i % 50 == 0:
                logger.info("Evaluating %d/%d", i + 1, len(examples))

            prediction_text = self.generate_answer(ex["question"])
            prediction = extract_answer(prediction_text)
            gold = ex["answer"]

            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            em_scores.append(em)
            f1_scores.append(f1)

            results.append({
                "id": ex.get("id", i),
                "question": ex["question"],
                "gold": gold,
                "prediction": prediction,
                "prediction_full": prediction_text,
                "em": em,
                "f1": f1,
            })

        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "em": avg_em,
            "f1": avg_f1,
            "num_examples": len(examples),
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-hop QA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset JSONL paths to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    evaluator = Evaluator(model_path=args.model_path)
    all_results = {}

    for dataset_path in args.datasets:
        name = Path(dataset_path).stem
        logger.info("Evaluating on %s", name)
        results = evaluator.evaluate_dataset(dataset_path, max_samples=args.max_samples)
        all_results[name] = {"em": results["em"], "f1": results["f1"], "n": results["num_examples"]}
        logger.info("  %s: EM=%.4f, F1=%.4f (n=%d)", name, results["em"], results["f1"], results["num_examples"])

    # Summary
    logger.info("=== Summary ===")
    for name, metrics in all_results.items():
        logger.info("  %s: EM=%.4f, F1=%.4f", name, metrics["em"], metrics["f1"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
