"""Prompt templates for search-reasoning trajectory generation.

Defines system prompts and few-shot examples that guide the teacher model
(Qwen3-32B) to generate interleaved think/search trajectories.

The trajectory format:
  <think>reasoning about what to search</think>
  <search>query</search>
  <results>...</results>
  <think>analyzing results, deciding next step</think>
  ...
  <think>synthesizing final answer</think>
  Answer: ...

References:
  - SimpleDeepSearcher (arXiv 2502.15975): think/search alternation
  - Search-R1 (arXiv 2503.09516): RL-trained search-reasoning
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a research assistant that answers complex questions by searching for \
information step by step.

## How to work
1. Wrap ALL your reasoning inside <think>...</think> tags.
2. When you need to look up a fact, emit <search>your query</search>.
   - You will then receive <results>...</results> containing retrieved passages.
3. After reading results, continue reasoning inside <think>...</think>.
4. Repeat search-then-think as many times as needed.
5. When you have enough information, write your final answer on a line \
starting with "Answer: ".

## Rules
- Every search query must be a short, specific keyword query (3-10 words).
- Do NOT fabricate facts. If search results are insufficient, search again \
with a different query.
- You may search up to 10 times. Use as few searches as needed.
- Always show your reasoning before and after each search."""

# ---------------------------------------------------------------------------
# User template (question placeholder)
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
Answer the following question by searching for information step by step.

Question: {question}"""

# ---------------------------------------------------------------------------
# Few-shot examples (2-hop and 3-hop)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # --- Example 1: 2-hop (bridge) ---
    {
        "role": "user",
        "content": (
            "Answer the following question by searching for information step by step.\n\n"
            "Question: What is the capital of the country where the Zambezi River ends?"
        ),
    },
    {
        "role": "assistant",
        "content": """\
<think>
I need to find where the Zambezi River ends (its mouth), then determine \
which country that is in, and finally find the capital of that country.

First, let me search for where the Zambezi River ends.
</think>
<search>Zambezi River mouth location</search>
<results>
[1] The Zambezi is the fourth-longest river in Africa. It rises in Zambia \
and flows through eastern Angola, Namibia, Botswana, Zambia, Zimbabwe, \
and Mozambique, emptying into the Indian Ocean near the city of Quelimane \
in Mozambique.
</results>
<think>
The Zambezi River empties into the Indian Ocean in Mozambique. So the \
country where it ends is Mozambique. Now I need the capital of Mozambique.
</think>
<search>capital of Mozambique</search>
<results>
[1] Maputo is the capital and largest city of Mozambique. It is located on \
the western shore of Maputo Bay.
</results>
<think>
The capital of Mozambique is Maputo. I have the complete chain: \
Zambezi ends in Mozambique, capital is Maputo.
</think>
Answer: Maputo""",
    },
    # --- Example 2: 3-hop (comparison) ---
    {
        "role": "user",
        "content": (
            "Answer the following question by searching for information step by step.\n\n"
            "Question: Were the directors of Jaws and E.T. born in the same city?"
        ),
    },
    {
        "role": "assistant",
        "content": """\
<think>
I need to find the directors of Jaws and E.T., then compare their \
birthplaces. Let me start with the director of Jaws.
</think>
<search>director of Jaws 1975 film</search>
<results>
[1] Jaws is a 1975 American thriller film directed by Steven Spielberg, \
based on Peter Benchley's 1974 novel.
</results>
<think>
Jaws was directed by Steven Spielberg. Now let me check the director of E.T.
</think>
<search>director of E.T. the Extra-Terrestrial</search>
<results>
[1] E.T. the Extra-Terrestrial is a 1982 American science fiction film \
produced and directed by Steven Spielberg.
</results>
<think>
Both Jaws and E.T. were directed by Steven Spielberg. So the question \
reduces to: was Steven Spielberg born in one city? That is trivially yes. \
Let me confirm his birthplace for completeness.
</think>
<search>Steven Spielberg birthplace</search>
<results>
[1] Steven Allan Spielberg was born on December 18, 1946, in Cincinnati, \
Ohio, United States.
</results>
<think>
Since both films were directed by the same person (Steven Spielberg, born \
in Cincinnati, Ohio), the answer is yes.
</think>
Answer: Yes, both Jaws and E.T. were directed by Steven Spielberg, who was born in Cincinnati, Ohio.""",
    },
]


def build_search_prompt(
    question: str,
    *,
    include_few_shot: bool = True,
) -> list[dict[str, str]]:
    """Build the chat-format prompt for trajectory generation.

    Args:
        question: The multi-hop question.
        include_few_shot: Whether to prepend few-shot examples.

    Returns:
        List of chat messages (role/content dicts).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if include_few_shot:
        messages.extend(FEW_SHOT_EXAMPLES)
    messages.append(
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    )
    return messages
