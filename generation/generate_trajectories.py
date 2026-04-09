"""Generate search-reasoning trajectories using a teacher model (Qwen3-32B).

Uses vLLM for efficient inference. The teacher generates interleaved
think→search→retrieve→reason trajectories for multi-hop QA questions.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Search-reasoning prompt template
SYSTEM_PROMPT = """You are a research assistant that answers questions by searching for information step by step.

When you need to find information, use <search>your query</search> to search.
After each search, you will receive relevant passages. Analyze them and decide whether to:
1. Search again with a refined query if you need more information
2. Synthesize your findings into a final answer

Think through each step carefully. Show your reasoning process."""

USER_TEMPLATE = """Answer the following question by searching for information step by step.

Question: {question}

Think step by step. Use <search>query</search> when you need to look up information."""


def build_search_prompt(question: str) -> list[dict[str, str]]:
    """Build the prompt for search-reasoning trajectory generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]


def inject_search_results(
    partial_response: str,
    search_engine: Any,
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    """Find search queries in the response and inject retrieved results.

    Args:
        partial_response: The model's partial generation containing <search>...</search>.
        search_engine: SearchEngine instance.
        top_k: Number of documents to retrieve per query.

    Returns:
        Tuple of (response_with_results, list_of_search_records).
    """
    search_records = []
    pattern = r"<search>(.*?)</search>"

    def replace_with_results(match: re.Match) -> str:
        query = match.group(1).strip()
        docs = search_engine.search(query, top_k=top_k)

        results_text = "\n".join(
            f"[{i+1}] {doc.to_context_string(max_chars=500)}"
            for i, doc in enumerate(docs)
        )

        search_records.append({
            "query": query,
            "num_results": len(docs),
            "doc_ids": [d.doc_id for d in docs],
        })

        return f"<search>{query}</search>\n<results>\n{results_text}\n</results>"

    augmented = re.sub(pattern, replace_with_results, partial_response, flags=re.DOTALL)
    return augmented, search_records


class TrajectoryGenerator:
    """Generate search-reasoning trajectories using vLLM.

    Implements iterative generation: generate → detect search query →
    retrieve results → continue generation with results injected.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-32B-AWQ",
        tensor_parallel_size: int = 2,
        max_model_len: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        max_search_steps: int = 10,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_search_steps = max_search_steps

        self.llm = None
        self.tokenizer = None
        self._init_vllm(tensor_parallel_size, max_model_len)

    def _init_vllm(self, tensor_parallel_size: int, max_model_len: int):
        """Initialize vLLM engine."""
        from vllm import LLM, SamplingParams

        logger.info("Initializing vLLM with %s (tp=%d)", self.model_name, tensor_parallel_size)
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            quantization="awq",
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["<search>"],  # Stop at search queries to inject results
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_trajectory(
        self,
        question: str,
        search_engine: Any,
        question_id: str = "",
    ) -> dict:
        """Generate a single search-reasoning trajectory.

        Uses iterative generation with search result injection.
        The stop token "<search>" is consumed by vLLM and NOT included in
        ``new_text``, so we must re-add it when constructing the query prompt.

        Args:
            question: The question to answer.
            search_engine: SearchEngine instance for retrieval.
            question_id: Identifier for the question.

        Returns:
            Dict with trajectory data including full text, search records, and metadata.
        """
        from vllm import SamplingParams

        messages = build_search_prompt(question)
        full_response = ""
        all_search_records = []
        search_step = 0
        incomplete = False
        start_time = time.time()

        while search_step < self.max_search_steps:
            # Generate until next search query or end
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt += full_response

            outputs = self.llm.generate([prompt], self.sampling_params)
            output = outputs[0].outputs[0]
            new_text = output.text

            # Determine why generation stopped
            stopped_at_search = output.stop_reason == "<search>"
            hit_max_tokens = (
                not stopped_at_search
                and len(output.token_ids) >= self.max_tokens
            )

            if stopped_at_search:
                # vLLM consumed "<search>" — it is NOT in new_text.
                # Continue generation to capture the query content up to </search>.
                query_prompt = prompt + new_text + "<search>"
                query_params = SamplingParams(
                    temperature=0.0,  # deterministic query
                    max_tokens=100,
                    stop=["</search>"],
                )
                query_outputs = self.llm.generate([query_prompt], query_params)
                query_text = query_outputs[0].outputs[0].text.strip()

                if not query_text:
                    # Model produced an empty query — treat as end of generation
                    full_response += new_text
                    logger.warning("Empty search query at step %d for %s", search_step, question_id)
                    break

                # Build the complete search block and inject retrieval results
                search_block = f"<search>{query_text}</search>"
                augmented, records = inject_search_results(
                    new_text + search_block, search_engine,
                )
                full_response += augmented
                all_search_records.extend(records)
                search_step += 1

            elif hit_max_tokens:
                # max_tokens exhausted mid-generation — mark trajectory incomplete
                full_response += new_text
                incomplete = True
                logger.warning(
                    "max_tokens reached at step %d for %s — marking incomplete",
                    search_step, question_id,
                )
                break

            else:
                # Normal EOS / natural stop — generation complete
                full_response += new_text
                break

        elapsed = time.time() - start_time
        return {
            "id": f"{question_id}_traj",
            "question_id": question_id,
            "question": question,
            "text": full_response,
            "search_records": all_search_records,
            "num_search_steps": search_step,
            "num_tokens": len(self.tokenizer.encode(full_response)),
            "generation_time_s": elapsed,
            "incomplete": incomplete,
        }

    def generate_batch(
        self,
        questions: list[dict],
        search_engine: Any,
        num_trajectories_per_question: int = 3,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Generate trajectories for a batch of questions.

        Args:
            questions: List of dicts with "id" and "question" keys.
            search_engine: SearchEngine instance.
            num_trajectories_per_question: Number of trajectories per question.
            output_path: If provided, write results incrementally to JSONL.

        Returns:
            List of trajectory dicts.
        """
        all_trajectories = []
        out_file = open(output_path, "w", encoding="utf-8") if output_path else None

        try:
            for qi, q in enumerate(questions):
                logger.info("Question %d/%d: %s", qi + 1, len(questions), q["question"][:80])
                for ti in range(num_trajectories_per_question):
                    traj = self.generate_trajectory(
                        question=q["question"],
                        search_engine=search_engine,
                        question_id=f"{q['id']}_{ti}",
                    )
                    traj["trajectory_idx"] = ti
                    all_trajectories.append(traj)

                    if out_file:
                        out_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                        out_file.flush()
        finally:
            if out_file:
                out_file.close()

        logger.info("Generated %d trajectories for %d questions", len(all_trajectories), len(questions))
        return all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Generate search-reasoning trajectories")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with questions")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL for trajectories")
    parser.add_argument("--corpus", type=str, required=True, help="Wikipedia corpus JSONL for BM25")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--num_traj", type=int, default=3, help="Trajectories per question")
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    # Load questions
    questions = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    if args.max_questions:
        questions = questions[:args.max_questions]

    # Initialize search engine
    from search_engine import BM25SearchEngine
    engine = BM25SearchEngine(corpus_path=args.corpus)

    # Generate trajectories
    generator = TrajectoryGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tp,
    )
    generator.generate_batch(
        questions=questions,
        search_engine=engine,
        num_trajectories_per_question=args.num_traj,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
