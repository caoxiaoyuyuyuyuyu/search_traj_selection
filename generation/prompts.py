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
   - The system will automatically retrieve relevant passages and return them \
inside <results>...</results>. You do NOT generate the results yourself.
3. After reading results, continue reasoning inside <think>...</think>.
4. Repeat search-then-think as many times as needed.
5. When you have enough information, write your final answer on a line \
starting with "Answer: ".

## Rules
- You MUST use <search>query</search> at least once before answering. \
Never answer from memory alone.
- Every search query must be a short, specific keyword query (3-10 words).
- Do NOT fabricate facts. If search results are insufficient, search again \
with a different query.
- You may search up to 10 times. Use as few searches as needed.
- Always show your reasoning before and after each search.
- Each question is INDEPENDENT. Do not reuse information from previous examples."""

# ---------------------------------------------------------------------------
# User template (question placeholder)
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
Answer the following NEW question by searching for information step by step. \
Do NOT reuse any information from previous examples.

Question: {question}"""

# ---------------------------------------------------------------------------
# Few-shot examples (2-hop and 3-hop)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # --- Example 1: 2-hop (bridge) ---
    {
        "role": "user",
        "content": (
            "Answer the following NEW question by searching for information step by step. Do NOT reuse any information from previous examples.\n\n"
            "Question: What is the capital of the country where the Zambezi River ends?"
        ),
    },
    {
        "role": "assistant",
        "content": """\
<think>
I need to find where the Zambezi River ends (its mouth), then determine \
which country that is in, and finally find the capital of that country. \
I don't know this from memory, so I must search for it.
</think>
<search>Zambezi River mouth location</search>
<results>
[1] The Zambezi is the fourth-longest river in Africa. It rises in Zambia \
and flows through eastern Angola, Namibia, Botswana, Zambia, Zimbabwe, \
and Mozambique, emptying into the Indian Ocean near the city of Quelimane \
in Mozambique.
</results>
<think>
The search results show that the Zambezi River empties into the Indian \
Ocean in Mozambique. So the country where it ends is Mozambique. Now I \
need to search for the capital of Mozambique.
</think>
<search>capital of Mozambique</search>
<results>
[1] Maputo is the capital and largest city of Mozambique. It is located on \
the western shore of Maputo Bay.
</results>
<think>
The search confirms that the capital of Mozambique is Maputo. I have the \
complete chain: Zambezi ends in Mozambique, capital is Maputo.
</think>
Answer: Maputo""",
    },
    # --- Example 2: 3-hop (comparison) — uses less well-known entities ---
    {
        "role": "user",
        "content": (
            "Answer the following NEW question by searching for information step by step. Do NOT reuse any information from previous examples.\n\n"
            "Question: Were the directors of Stalker and Nostalghia born in the same country?"
        ),
    },
    {
        "role": "assistant",
        "content": """\
<think>
I need to find the directors of these two films, then check if they were \
born in the same country. I should not guess — let me search for each.
</think>
<search>director of Stalker 1979 film</search>
<results>
[1] Stalker is a 1979 Soviet science fiction art film directed by Andrei \
Tarkovsky, loosely based on the novel Roadside Picnic by Arkady and Boris \
Strugatsky.
</results>
<think>
Stalker was directed by Andrei Tarkovsky. Now I need to search for the \
director of Nostalghia.
</think>
<search>director of Nostalghia 1983 film</search>
<results>
[1] Nostalghia is a 1983 art film directed by Andrei Tarkovsky. It was \
Tarkovsky's first film made outside the Soviet Union, produced in Italy.
</results>
<think>
Both films were directed by Andrei Tarkovsky. Since it's the same person, \
they were obviously born in the same country. But let me confirm his \
birthplace with a search to give a complete answer.
</think>
<search>Andrei Tarkovsky birthplace</search>
<results>
[1] Andrei Arsenyevich Tarkovsky was born on 4 April 1932 in Zavrazhye, \
Yuryevetsky District, Ivanovo Industrial Oblast, Soviet Union (now Russia).
</results>
<think>
Confirmed: both films were directed by Andrei Tarkovsky, born in the \
Soviet Union (now Russia). The answer is yes.
</think>
Answer: Yes, both Stalker and Nostalghia were directed by Andrei Tarkovsky, who was born in the Soviet Union (Zavrazhye, now Russia).""",
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
