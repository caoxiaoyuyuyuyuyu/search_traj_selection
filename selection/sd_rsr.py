"""SD-RSR: Search-Decision RSR — our proposed method.

Addresses signal dilution in vanilla RSR for search-reasoning trajectories:
search query tokens constitute only ~5-15% of trajectory tokens, so vanilla RSR
is dominated by reasoning tokens and blind to search strategy quality.

SD-RSR decomposes RSR into two components:
  SD-RSR(T, S, α) = (1 - α) · RSR_reasoning(T, S) + α · RSR_search_decision(T, S)

where:
  - RSR_reasoning = RSR computed over reasoning tokens only
  - RSR_search_decision = RSR computed over search-decision tokens only
    (query formulation + search-or-continue decision tokens)
  - α = tunable weight capturing the student's relative skill gap
"""

import logging
import re
from dataclasses import dataclass

import torch
from torch import Tensor

from .rsr import RSRComputer, RSRResult

logger = logging.getLogger(__name__)


@dataclass
class SDRSRResult:
    """Extended RSR result with decomposed scores."""
    trajectory_id: str
    sd_rsr_score: float  # weighted composite
    rsr_reasoning: float  # RSR on reasoning tokens only
    rsr_search_decision: float  # RSR on search-decision tokens only
    alpha: float  # weight used
    num_reasoning_tokens: int
    num_search_decision_tokens: int
    search_decision_ratio: float  # fraction of tokens that are search-decision
    # Original full-trajectory RSR for comparison
    rsr_full: float


class SearchDecisionTokenizer:
    """Identify search-decision token spans in trajectory text.

    Search-decision tokens include:
    1. Query formulation tokens: text within <search>...</search> tags
    2. Decision markers: tokens immediately surrounding search/continue/synthesize decisions
    3. (Extended mode) Pre-search reasoning: sentences in <think> blocks that express
       intent to search (e.g., "let me search for...", "I need to find...")

    The specific markers depend on the trajectory format. This class handles
    the common format used by search-reasoning systems (Search-R1, SimpleDeepSearcher, etc.).
    """

    # Patterns that indicate search-decision reasoning within <think> blocks
    SEARCH_INTENT_PATTERNS = [
        r"[Ll]et me search\b",
        r"[Ll]et me look up\b",
        r"[Ll]et me find\b",
        r"[Ii] need to (?:search|find|look up)\b",
        r"[Ii] should (?:search|find|look up)\b",
        r"[Ii] (?:will|'ll) search\b",
        r"[Nn]ow (?:let me |I (?:need to |will |should ))?(?:search|find|look up)\b",
        r"[Ff]irst,? (?:let me |I (?:need to |will |should ))?(?:search|find|look up)\b",
        r"[Nn]ext,? (?:let me |I (?:need to |will |should ))?(?:search|find|look up)\b",
    ]

    def __init__(
        self,
        query_start: str = "<search>",
        query_end: str = "</search>",
        decision_markers: list[str] | None = None,
        extended: bool = False,
    ):
        self.query_start = query_start
        self.query_end = query_end
        self.decision_markers = decision_markers or ["<search>", "<continue>", "<synthesize>"]
        self.extended = extended
        self._intent_re = re.compile("|".join(self.SEARCH_INTENT_PATTERNS))

    def find_search_decision_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Find character-level spans of search-decision tokens.

        Returns:
            List of (start_char, end_char, "search_decision") tuples.
            All other characters are implicitly "reasoning".
        """
        spans: list[tuple[int, int, str]] = []

        # Find query content spans: text between query_start and query_end
        pattern = re.escape(self.query_start) + r"(.*?)" + re.escape(self.query_end)
        for match in re.finditer(pattern, text, re.DOTALL):
            # Include the tags themselves as search-decision tokens
            spans.append((match.start(), match.end(), "search_decision"))

        # Find standalone decision markers not already covered
        for marker in self.decision_markers:
            for match in re.finditer(re.escape(marker), text):
                # Check overlap with existing spans
                if not any(s[0] <= match.start() and match.end() <= s[1] for s in spans):
                    spans.append((match.start(), match.end(), "search_decision"))

        # Extended mode: find search-intent sentences in <think> blocks
        if self.extended:
            for think_match in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
                think_text = think_match.group(1)
                think_start = think_match.start(1)
                # Split into sentences and check each for search intent
                for sent_match in re.finditer(r"[^.!?\n]+[.!?\n]?", think_text):
                    sentence = sent_match.group()
                    if self._intent_re.search(sentence):
                        abs_start = think_start + sent_match.start()
                        abs_end = think_start + sent_match.end()
                        # Check overlap with existing spans
                        if not any(s[0] <= abs_start and abs_end <= s[1] for s in spans):
                            spans.append((abs_start, abs_end, "search_decision"))

        # Sort by start position
        spans.sort(key=lambda s: s[0])
        return spans


class SDRSRComputer:
    """Compute SD-RSR scores for search-reasoning trajectories.

    Usage:
        computer = SDRSRComputer(
            student_model_name="Qwen/Qwen3-8B",
            alpha=0.6,
        )
        results = computer.score_trajectories(trajectories)
    """

    def __init__(
        self,
        student_model_name: str,
        alpha: float = 0.5,
        r_max: int = 100,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        query_start: str = "<search>",
        query_end: str = "</search>",
        decision_markers: list[str] | None = None,
    ):
        self.alpha = alpha
        self.rsr_computer = RSRComputer(
            student_model_name=student_model_name,
            r_max=r_max,
            device=device,
            torch_dtype=torch_dtype,
            max_length=max_length,
        )
        self.sd_tokenizer = SearchDecisionTokenizer(
            query_start=query_start,
            query_end=query_end,
            decision_markers=decision_markers,
        )

    def score_trajectory(
        self,
        trajectory_id: str,
        text: str,
        alpha: float | None = None,
    ) -> SDRSRResult:
        """Score a single trajectory with SD-RSR decomposition.

        Args:
            trajectory_id: Unique identifier.
            text: Full trajectory text.
            alpha: Override instance-level alpha for this trajectory.

        Returns:
            SDRSRResult with decomposed scores.
        """
        alpha = alpha if alpha is not None else self.alpha

        # Find search-decision spans
        sd_spans = self.sd_tokenizer.find_search_decision_spans(text)

        # Get full RSR result with token details
        full_result = self.rsr_computer.score_trajectory(
            trajectory_id=trajectory_id,
            text=text,
            token_type_spans=sd_spans,
            return_token_details=True,
        )

        # Build masks for reasoning vs search-decision tokens
        token_types = full_result.token_types or ["reasoning"] * full_result.num_tokens
        reasoning_mask = torch.tensor([t == "reasoning" for t in token_types], dtype=torch.bool)
        sd_mask = torch.tensor([t == "search_decision" for t in token_types], dtype=torch.bool)

        ranks = torch.tensor(full_result.token_ranks, dtype=torch.float32)
        surprisals = torch.tensor(full_result.token_surprisals, dtype=torch.float32)

        # Compute decomposed RSR
        rsr_reasoning = self.rsr_computer.compute_rsr(ranks.long(), surprisals, mask=reasoning_mask)
        rsr_search_decision = self.rsr_computer.compute_rsr(ranks.long(), surprisals, mask=sd_mask)

        # Weighted composite
        # Handle inf values: if one component is inf, use the other
        if rsr_search_decision == float("inf") and rsr_reasoning == float("inf"):
            sd_rsr_score = float("inf")
        elif rsr_search_decision == float("inf"):
            sd_rsr_score = rsr_reasoning  # fall back to reasoning-only
        elif rsr_reasoning == float("inf"):
            sd_rsr_score = rsr_search_decision  # fall back to search-only
        else:
            sd_rsr_score = (1 - alpha) * rsr_reasoning + alpha * rsr_search_decision

        num_sd = int(sd_mask.sum().item())
        num_reasoning = int(reasoning_mask.sum().item())
        total = num_sd + num_reasoning

        return SDRSRResult(
            trajectory_id=trajectory_id,
            sd_rsr_score=sd_rsr_score,
            rsr_reasoning=rsr_reasoning,
            rsr_search_decision=rsr_search_decision,
            alpha=alpha,
            num_reasoning_tokens=num_reasoning,
            num_search_decision_tokens=num_sd,
            search_decision_ratio=num_sd / total if total > 0 else 0.0,
            rsr_full=full_result.rsr_score,
        )

    def score_trajectories(
        self,
        trajectories: list[dict],
        alpha: float | None = None,
    ) -> list[SDRSRResult]:
        """Score multiple trajectories.

        Args:
            trajectories: List of dicts with keys: id, text.
            alpha: Override instance-level alpha.

        Returns:
            List of SDRSRResult, sorted by sd_rsr_score ascending.
        """
        results = []
        for i, traj in enumerate(trajectories):
            if i % 50 == 0:
                logger.info("SD-RSR scoring trajectory %d/%d", i + 1, len(trajectories))
            result = self.score_trajectory(
                trajectory_id=traj["id"],
                text=traj["text"],
                alpha=alpha,
            )
            results.append(result)

        results.sort(key=lambda r: r.sd_rsr_score)
        return results

    def alpha_sweep(
        self,
        trajectories: list[dict],
        alpha_values: list[float] | None = None,
    ) -> dict[float, list[SDRSRResult]]:
        """Run SD-RSR scoring across multiple alpha values.

        Efficiently reuses cached ranks/surprisals (computed once per trajectory).

        Args:
            trajectories: List of trajectory dicts.
            alpha_values: Alpha values to sweep. Default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].

        Returns:
            Dict mapping alpha → sorted list of SDRSRResult.
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Pre-compute token ranks and surprisals for all trajectories
        logger.info("Pre-computing ranks/surprisals for %d trajectories", len(trajectories))
        cached: list[dict] = []
        for i, traj in enumerate(trajectories):
            if i % 50 == 0:
                logger.info("  Pre-computing %d/%d", i + 1, len(trajectories))

            sd_spans = self.sd_tokenizer.find_search_decision_spans(traj["text"])
            full_result = self.rsr_computer.score_trajectory(
                trajectory_id=traj["id"],
                text=traj["text"],
                token_type_spans=sd_spans,
                return_token_details=True,
            )
            cached.append({
                "id": traj["id"],
                "full_result": full_result,
                "sd_spans": sd_spans,
            })

        # Sweep alpha values using cached data
        sweep_results: dict[float, list[SDRSRResult]] = {}
        for alpha in alpha_values:
            logger.info("Alpha sweep: α=%.1f", alpha)
            results = []
            for item in cached:
                fr = item["full_result"]
                token_types = fr.token_types or ["reasoning"] * fr.num_tokens
                reasoning_mask = torch.tensor([t == "reasoning" for t in token_types], dtype=torch.bool)
                sd_mask = torch.tensor([t == "search_decision" for t in token_types], dtype=torch.bool)

                ranks = torch.tensor(fr.token_ranks, dtype=torch.long)
                surprisals = torch.tensor(fr.token_surprisals, dtype=torch.float32)

                rsr_r = self.rsr_computer.compute_rsr(ranks, surprisals, mask=reasoning_mask)
                rsr_sd = self.rsr_computer.compute_rsr(ranks, surprisals, mask=sd_mask)

                if rsr_sd == float("inf") and rsr_r == float("inf"):
                    score = float("inf")
                elif rsr_sd == float("inf"):
                    score = rsr_r
                elif rsr_r == float("inf"):
                    score = rsr_sd
                else:
                    score = (1 - alpha) * rsr_r + alpha * rsr_sd

                num_sd = int(sd_mask.sum().item())
                num_r = int(reasoning_mask.sum().item())
                total = num_sd + num_r

                results.append(SDRSRResult(
                    trajectory_id=item["id"],
                    sd_rsr_score=score,
                    rsr_reasoning=rsr_r,
                    rsr_search_decision=rsr_sd,
                    alpha=alpha,
                    num_reasoning_tokens=num_r,
                    num_search_decision_tokens=num_sd,
                    search_decision_ratio=num_sd / total if total > 0 else 0.0,
                    rsr_full=fr.rsr_score,
                ))

            results.sort(key=lambda r: r.sd_rsr_score)
            sweep_results[alpha] = results

        return sweep_results
