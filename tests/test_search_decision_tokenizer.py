"""Verify SearchDecisionTokenizer on realistic search-reasoning trajectories.

Tests that the tokenizer correctly identifies search-decision spans in
trajectories using the <think>/<search> format from prompts.py.
"""

import re
import sys
from pathlib import Path


# Inline SearchDecisionTokenizer to avoid torch dependency in local tests
class SearchDecisionTokenizer:
    """Copy of selection.sd_rsr.SearchDecisionTokenizer for offline testing."""

    def __init__(
        self,
        query_start: str = "<search>",
        query_end: str = "</search>",
        decision_markers: list | None = None,
    ):
        self.query_start = query_start
        self.query_end = query_end
        self.decision_markers = decision_markers or ["<search>", "<continue>", "<synthesize>"]

    def find_search_decision_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        pattern = re.escape(self.query_start) + r"(.*?)" + re.escape(self.query_end)
        for match in re.finditer(pattern, text, re.DOTALL):
            spans.append((match.start(), match.end(), "search_decision"))
        for marker in self.decision_markers:
            for match in re.finditer(re.escape(marker), text):
                if not any(s[0] <= match.start() and match.end() <= s[1] for s in spans):
                    spans.append((match.start(), match.end(), "search_decision"))
        spans.sort(key=lambda s: s[0])
        return spans


# ---------------------------------------------------------------------------
# Realistic mock trajectories (mimicking Qwen3-32B output on HotpotQA)
# ---------------------------------------------------------------------------

# Trajectory 1: 2-hop bridge question
TRAJ_BRIDGE = """\
<think>
The question asks about the nationality of the director of "Spirited Away". \
I need to first find who directed Spirited Away, then determine their nationality.
</think>
<search>director of Spirited Away animated film</search>
<results>
[1] Spirited Away is a 2001 Japanese animated fantasy film written and \
directed by Hayao Miyazaki, produced by Studio Ghibli.
[2] The film won the Academy Award for Best Animated Feature in 2003.
</results>
<think>
Spirited Away was directed by Hayao Miyazaki. Now I need to find his nationality.
</think>
<search>Hayao Miyazaki nationality birthplace</search>
<results>
[1] Hayao Miyazaki is a Japanese animator, filmmaker, and manga artist. \
He was born on January 5, 1941, in Bunkyo, Tokyo, Japan.
</results>
<think>
Hayao Miyazaki is Japanese. So the director of Spirited Away is Japanese.
</think>
Answer: Japanese"""

# Trajectory 2: 3-hop comparison question (more search steps)
TRAJ_COMPARISON = """\
<think>
I need to find the birth years of the authors of "1984" and "Brave New World" \
and determine who was born first.
</think>
<search>author of 1984 novel</search>
<results>
[1] Nineteen Eighty-Four (1984) is a dystopian novel by English author \
George Orwell, published in 1949.
</results>
<think>
George Orwell wrote 1984. Now I need the author of Brave New World.
</think>
<search>author of Brave New World</search>
<results>
[1] Brave New World is a dystopian novel by English author Aldous Huxley, \
published in 1932.
</results>
<think>
Aldous Huxley wrote Brave New World. Now I need both birth years.
</think>
<search>George Orwell birth year</search>
<results>
[1] Eric Arthur Blair, known by his pen name George Orwell, was born on \
25 June 1903 in Motihari, Bengal Presidency, British India.
</results>
<search>Aldous Huxley birth year</search>
<results>
[1] Aldous Leonard Huxley was born on 26 July 1894 in Godalming, Surrey, England.
</results>
<think>
George Orwell was born in 1903, Aldous Huxley in 1894. Huxley was born first.
</think>
Answer: Aldous Huxley was born first (1894 vs 1903)."""

# Trajectory 3: single search (edge case — minimal search-decision tokens)
TRAJ_SINGLE = """\
<think>
This is a straightforward factual question. Let me search directly.
</think>
<search>population of Tokyo 2023</search>
<results>
[1] Tokyo has an estimated population of 13.96 million as of 2023.
</results>
<think>
The population of Tokyo is approximately 13.96 million.
</think>
Answer: Approximately 13.96 million"""


def test_span_detection():
    """Verify spans are detected correctly on all three trajectories."""
    tokenizer = SearchDecisionTokenizer()

    print("=" * 60)
    print("Test 1: 2-hop bridge trajectory")
    print("=" * 60)
    spans = tokenizer.find_search_decision_spans(TRAJ_BRIDGE)
    print(f"  Found {len(spans)} search-decision spans:")
    for start, end, label in spans:
        snippet = TRAJ_BRIDGE[start:end]
        print(f"  [{start}:{end}] {label}: {snippet!r}")
    # Expect 2 <search>...</search> spans
    search_spans = [s for s in spans if "<search>" in TRAJ_BRIDGE[s[0]:s[1]]]
    assert len(search_spans) == 2, f"Expected 2 search spans, got {len(search_spans)}"
    # Verify query content is inside spans
    assert "director of Spirited Away" in TRAJ_BRIDGE[search_spans[0][0]:search_spans[0][1]]
    assert "Hayao Miyazaki nationality" in TRAJ_BRIDGE[search_spans[1][0]:search_spans[1][1]]
    print("  PASS\n")

    print("=" * 60)
    print("Test 2: 3-hop comparison trajectory")
    print("=" * 60)
    spans = tokenizer.find_search_decision_spans(TRAJ_COMPARISON)
    print(f"  Found {len(spans)} search-decision spans:")
    for start, end, label in spans:
        snippet = TRAJ_COMPARISON[start:end]
        print(f"  [{start}:{end}] {label}: {snippet!r}")
    search_spans = [s for s in spans if "<search>" in TRAJ_COMPARISON[s[0]:s[1]]]
    assert len(search_spans) == 4, f"Expected 4 search spans, got {len(search_spans)}"
    print("  PASS\n")

    print("=" * 60)
    print("Test 3: Single-search trajectory")
    print("=" * 60)
    spans = tokenizer.find_search_decision_spans(TRAJ_SINGLE)
    print(f"  Found {len(spans)} search-decision spans:")
    for start, end, label in spans:
        snippet = TRAJ_SINGLE[start:end]
        print(f"  [{start}:{end}] {label}: {snippet!r}")
    search_spans = [s for s in spans if "<search>" in TRAJ_SINGLE[s[0]:s[1]]]
    assert len(search_spans) == 1, f"Expected 1 search span, got {len(search_spans)}"
    print("  PASS\n")


def test_search_decision_ratio():
    """Verify search-decision token ratios are in expected range (5-15%)."""
    tokenizer = SearchDecisionTokenizer()

    for name, traj in [("bridge", TRAJ_BRIDGE), ("comparison", TRAJ_COMPARISON), ("single", TRAJ_SINGLE)]:
        spans = tokenizer.find_search_decision_spans(traj)
        sd_chars = sum(end - start for start, end, _ in spans)
        total_chars = len(traj)
        ratio = sd_chars / total_chars
        print(f"  {name}: {sd_chars}/{total_chars} chars = {ratio:.1%} search-decision")
        # Search-decision should be a minority of the trajectory
        assert ratio < 0.30, f"Search-decision ratio too high: {ratio:.1%}"
        assert ratio > 0.02, f"Search-decision ratio too low: {ratio:.1%}"

    print("  PASS — all ratios in expected range\n")


def test_no_overlap():
    """Verify spans do not overlap."""
    tokenizer = SearchDecisionTokenizer()

    for traj in [TRAJ_BRIDGE, TRAJ_COMPARISON, TRAJ_SINGLE]:
        spans = tokenizer.find_search_decision_spans(traj)
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0], (
                f"Overlap: span {spans[i]} overlaps with {spans[i+1]}"
            )
    print("  PASS — no overlapping spans\n")


if __name__ == "__main__":
    test_span_detection()

    print("=" * 60)
    print("Test 4: Search-decision ratios")
    print("=" * 60)
    test_search_decision_ratio()

    print("=" * 60)
    print("Test 5: No overlapping spans")
    print("=" * 60)
    test_no_overlap()

    print("ALL TESTS PASSED")
