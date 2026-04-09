"""RSR (Rank-Surprisal Ratio) baseline implementation.

Reference: arXiv 2601.14249 — "Which Reasoning Trajectories Teach Students to Reason Better?"

RSR(T, S) = Σ min(Rank_S(t_k), r_max) / Σ Surprisal_S(t_k)

where:
  - T = trajectory token sequence
  - S = student model
  - Rank_S(t_k) = rank of token t_k in student's vocab distribution at position k
  - Surprisal_S(t_k) = -log P_S(t_k | t_{<k})
  - r_max = rank clipping threshold (default 100)

Lower RSR → trajectory is more informative for the student (high-rank but low-surprisal
tokens indicate the student is learning new patterns that fit its distribution).
"""

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RSRResult:
    """Result of RSR computation for a single trajectory."""
    trajectory_id: str
    rsr_score: float
    num_tokens: int
    mean_rank: float
    mean_surprisal: float
    # Per-token details (optional, for analysis)
    token_ranks: list[int] | None = None
    token_surprisals: list[float] | None = None
    token_types: list[str] | None = None  # "reasoning" | "search_decision" for SD-RSR


class RSRComputer:
    """Compute RSR scores for trajectories given a student model.

    Usage:
        computer = RSRComputer(student_model_name="Qwen/Qwen3-8B", r_max=100)
        results = computer.score_trajectories(trajectories)
    """

    def __init__(
        self,
        student_model_name: str,
        r_max: int = 100,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ):
        self.r_max = r_max
        self.device = device
        self.max_length = max_length

        logger.info("Loading student model: %s", student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.no_grad()
    def compute_token_ranks_and_surprisals(
        self, input_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute per-token rank and surprisal for a single sequence.

        Args:
            input_ids: (1, seq_len) token IDs.

        Returns:
            ranks: (seq_len-1,) rank of each token in the student's distribution.
            surprisals: (seq_len-1,) -log P(t_k | t_{<k}).
        """
        outputs = self.model(input_ids)
        # logits: (1, seq_len, vocab_size) — shift to align with targets
        logits = outputs.logits[0, :-1, :]  # (seq_len-1, vocab_size)
        targets = input_ids[0, 1:]  # (seq_len-1,)

        # Surprisal: -log P(t_k | t_{<k})
        log_probs = F.log_softmax(logits, dim=-1)  # (seq_len-1, vocab_size)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (seq_len-1,)
        surprisals = -token_log_probs  # (seq_len-1,)

        # Rank: position of target token when sorted by descending probability
        # rank 0 = most likely token
        sorted_indices = logits.argsort(dim=-1, descending=True)  # (seq_len-1, vocab_size)
        ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]  # (seq_len-1,)

        return ranks, surprisals

    def compute_rsr(
        self,
        ranks: Tensor,
        surprisals: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute RSR from pre-computed ranks and surprisals.

        Args:
            ranks: (N,) per-token ranks.
            surprisals: (N,) per-token surprisals.
            mask: (N,) optional boolean mask to select specific tokens.

        Returns:
            RSR score (lower = more informative for student).
        """
        if mask is not None:
            ranks = ranks[mask]
            surprisals = surprisals[mask]

        if len(ranks) == 0:
            return float("inf")

        clipped_ranks = torch.clamp(ranks.float(), max=self.r_max)
        total_rank = clipped_ranks.sum().item()
        total_surprisal = surprisals.sum().item()

        if total_surprisal < 1e-8:
            return float("inf")

        return total_rank / total_surprisal

    def score_trajectory(
        self,
        trajectory_id: str,
        text: str,
        token_type_spans: list[tuple[int, int, str]] | None = None,
        return_token_details: bool = False,
    ) -> RSRResult:
        """Score a single trajectory.

        Args:
            trajectory_id: Unique identifier for this trajectory.
            text: Full trajectory text.
            token_type_spans: Optional list of (start_char, end_char, type) for
                token type annotation. type ∈ {"reasoning", "search_decision"}.
            return_token_details: If True, include per-token ranks/surprisals.

        Returns:
            RSRResult with the computed RSR score.
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offset_mapping = encoded["offset_mapping"][0]  # (seq_len, 2)

        ranks, surprisals = self.compute_token_ranks_and_surprisals(input_ids)

        # Build token type labels if spans provided
        token_types = None
        if token_type_spans is not None:
            token_types = self._assign_token_types(offset_mapping[1:], token_type_spans)

        rsr_score = self.compute_rsr(ranks, surprisals)

        result = RSRResult(
            trajectory_id=trajectory_id,
            rsr_score=rsr_score,
            num_tokens=len(ranks),
            mean_rank=ranks.float().mean().item(),
            mean_surprisal=surprisals.mean().item(),
        )

        if return_token_details:
            result.token_ranks = ranks.cpu().tolist()
            result.token_surprisals = surprisals.cpu().tolist()
            result.token_types = token_types

        return result

    def score_trajectories(
        self,
        trajectories: list[dict],
        batch_size: int = 1,
        return_token_details: bool = False,
    ) -> list[RSRResult]:
        """Score multiple trajectories.

        Args:
            trajectories: List of dicts with keys:
                - id: trajectory identifier
                - text: trajectory text
                - token_type_spans: (optional) list of (start, end, type)
            batch_size: Currently processes one at a time (due to variable lengths).
            return_token_details: Include per-token details in results.

        Returns:
            List of RSRResult, sorted by RSR score (ascending = most informative first).
        """
        results = []
        for i, traj in enumerate(trajectories):
            if i % 50 == 0:
                logger.info("Scoring trajectory %d/%d", i + 1, len(trajectories))
            result = self.score_trajectory(
                trajectory_id=traj["id"],
                text=traj["text"],
                token_type_spans=traj.get("token_type_spans"),
                return_token_details=return_token_details,
            )
            results.append(result)

        results.sort(key=lambda r: r.rsr_score)
        return results

    @staticmethod
    def _assign_token_types(
        offsets: Tensor,
        spans: list[tuple[int, int, str]],
    ) -> list[str]:
        """Assign token types based on character-level spans.

        Args:
            offsets: (num_tokens, 2) character offset mapping for each token.
            spans: List of (start_char, end_char, type_label).

        Returns:
            List of type labels, one per token. Default: "reasoning".
        """
        types = ["reasoning"] * len(offsets)
        for tok_idx in range(len(offsets)):
            tok_start, tok_end = offsets[tok_idx].tolist()
            for span_start, span_end, label in spans:
                if tok_start >= span_start and tok_end <= span_end:
                    types[tok_idx] = label
                    break
        return types


def select_top_k(
    results: Sequence[RSRResult],
    k: int,
    lower_is_better: bool = True,
) -> list[RSRResult]:
    """Select top-k trajectories by RSR score.

    Args:
        results: RSR results to select from.
        k: Number of trajectories to select.
        lower_is_better: If True (default for RSR), select lowest scores.

    Returns:
        Top-k RSRResult objects.
    """
    sorted_results = sorted(results, key=lambda r: r.rsr_score, reverse=not lower_is_better)
    return list(sorted_results[:k])
