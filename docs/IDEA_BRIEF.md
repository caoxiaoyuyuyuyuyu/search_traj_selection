# Idea Brief: Student-Calibrated Trajectory Selection for Search-Reasoning Distillation


- **Idea ID**: `idea_026`
- **Created**: 2026-04-08
- **Approved**: 2026-04-09
- **Source**: web_research
- **Level**: top-tier
- **Target Venues**: EMNLP 2026 (ARR submission May 25 — 47 days, achievable with focused scope and 2×RTX PRO 6000)

## Seed Topic

帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！

## Generation Journey

1. **Start** — 开始生成，seed: '帮我生成和LLM agent、harness engineering、LLM+搜索、LLM+graphics相关的idea，要求能中顶会，用双卡RTX PRO 6000（单卡96GB显存）及以下的资源（GPU 100h以下），尽量不用API key，不自己构建benchmark，不需要人工标注，尽量不要做RL训练（显存占用过高）！'
2. **Problem Discovery** (714s) — 发现 4 个研究方向
   - SFT Distillation of Interleaved Search-Reasoning for Small LLMs
   - Rendering-Grounded Rejection Sampling for Visual Code SFT
   - Learned Online Trajectory Compression for LLM Agents
   - Multi-Format Visual Code Refinement via Difference-Aligned SFT
3. **Method Synthesis** (281s) — 完成: "SFT Distillation of Interleaved Search-Reasoning for Small LLMs" [solid] → NeurIPS 2026 (abstract May 4, full paper May 6), EMNLP 2026 (ARR submission May 25)
3. **Method Synthesis** (182s) — 完成: "Rendering-Grounded Rejection Sampling for Visual Code SFT" [solid] → NeurIPS 2026 (paper deadline May 6, 2026), EMNLP 2026 (ARR deadline May 25, 2026)
3. **Method Synthesis** (393s) — 完成: "Learned Online Trajectory Compression for LLM Agents" [solid] → NeurIPS 2026 (abstract May 4, paper May 6, 2026), EMNLP 2026 (ARR submission May 25, 2026; commitment Aug 2, 2026)
3. **Method Synthesis** (426s) — 完成: "Multi-Format Visual Code Refinement via Difference-Aligned SFT" [solid] → NeurIPS 2026 (abstract May 4, paper May 6), EMNLP 2026 (ARR submission May 25)

## Core Problem

SFT distillation of search-reasoning from large to small LLMs is established (SimpleDeepSearcher, OpenSeeker), but trajectory selection remains student-agnostic — filtered by answer correctness or uniform quality criteria. Recent work proves trajectory-student compatibility critically determines distillation quality for math reasoning (RSR achieves 0.86 Spearman; Gen-SSD improves ~5.9 points). However, RSR was designed and evaluated exclusively on math CoT. Search-reasoning trajectories have a fundamentally different structure — interleaved think→query→retrieve→reason steps — that creates a specific mechanistic failure mode for token-level metrics like RSR: RSR captures token-level learnability but is blind to sequential search strategy quality. A trajectory where the student can predict each individual query token (low RSR, seemingly learnable) but would never generate the correct sequence of queries (poor strategic search planning) receives a misleadingly good RSR score. This decoupling of token-level and strategy-level learnability is unique to search-reasoning — in math CoT, token-level prediction ability and step-level reasoning quality are tightly coupled because each step is self-contained. In search-reasoning, query formulation depends on what was retrieved previously, creating cross-step dependencies that token-level metrics miss.

## Opportunity / Why Now

RSR's authors explicitly state their metric 'is not specifically designed for reasoning trajectories and can be applied to general text data' but acknowledge extension beyond math as 'an interesting direction for future work' (arXiv 2601.14249). This creates an author-acknowledged gap. Meanwhile, all search-reasoning trajectory quality approaches — OpenSeeker's retrospective summarization, SimpleDeepSearcher's four-dimensional curation, OpenResearcher's 97K trajectory pipeline — apply uniform quality filters regardless of which student will learn from them. Gen-SSD and Skill-Aware Selection (arXiv 2601.10109) showed that the same trajectory can be excellent for one student and useless for another in math reasoning, and that targeting student-specific skill weaknesses improves data efficiency. We hypothesize this effect is amplified in search-reasoning because students fail in qualitatively different ways along two orthogonal skill axes: a student might already reason well but search poorly (needs trajectories emphasizing search strategy), or search adequately but reason poorly over retrieved content (needs trajectories emphasizing synthesis). A smaller student (4B) may struggle with both, while a larger student (8B) may only need search skill improvement. One-size-fits-all filtering cannot capture this student-dependent skill gap structure.

## Landscape (SOTA & Limitations)

**Search-reasoning distillation:** Search-R1 (COLM 2025) uses RL with retrieved-token masking. AceSearcher (NeurIPS 2025 Spotlight) bootstraps via cooperative self-play SFT+RL. SimpleDeepSearcher (EMNLP 2025 Findings) achieves strong results with SFT on only 871 curated trajectories using four-dimensional quality filtering. OpenSeeker (arXiv 2603.15594) scales to 11.7K denoised trajectories via retrospective summarization. HybridDeepSearcher (ICLR 2026) adds structural trajectory diversity. OpenResearcher (arXiv 2603.20278) provides 97K+ long-horizon search trajectories with 100+ tool calls, showing SFT alone unlocks significant deep-research gains. **Trajectory selection for distillation:** RSR (arXiv 2601.14249) proposes Rank-Surprisal Ratio measuring token-level informative alignment, achieving 0.86 Spearman on math but only evaluated on math CoT. Gen-SSD (arXiv 2604.02819) introduces student-in-the-loop generation-time selection, +5.9 over standard KD on math. DASD (arXiv 2601.09088) addresses distribution-level teacher-student mismatch via divergence-aware sampling and temperature scheduling, achieving SOTA with 448K samples. T3S (arXiv 2601.10348) discovers token-level bifurcation into imitation-anchor vs yet-to-learn tokens, reconstructing training objectives at token level. Skill-Aware Selection (arXiv 2601.10109) targets student's weaker skills for data-efficient distillation, +1.6% on Qwen3-4B with only 1K examples. **Agent trajectory selection:** TopoCurate (arXiv 2603.01714) proposes topology-based dual-selection for tool-use agent trajectories, selecting for reflective recovery, semantic efficiency, and diversity — student-agnostic but structurally aware. 'What Do Agents Learn from Trajectory-SFT' (arXiv 2602.01611) reveals that trajectory-SFT can teach interface shortcuts rather than semantic tool-use, with trained agents degrading sharply under interface changes — raising the question of whether search-reasoning SFT teaches genuine search strategy or surface patterns. Self-distillation degradation work (arXiv 2603.24472) warns that suppressing epistemic verbalization degrades reasoning by up to 40%. **The gap:** All trajectory selection work targets pure reasoning chains (math/code). Agent trajectory selection (TopoCurate) considers structural quality but not student capacity. No work has studied whether student-calibrated selection transfers to search-reasoning, where the token-level vs strategy-level learnability decoupling creates a unique challenge.

## Proposed Approach

We study whether student-calibrated trajectory selection transfers from math reasoning to search-reasoning distillation, testing a specific mechanistic hypothesis about WHY token-level metrics fail for search-reasoning, and proposing a search-aware decomposition to address this.

**Setup:** Teacher Qwen3-32B (AWQ-INT4 via vLLM, ~30-50 tok/s on RTX PRO 6000) generates search-reasoning trajectories on multi-hop QA (HotpotQA, MuSiQue, 2WikiMHQA) using local BM25+ColBERT retrieval over cached Wikipedia. Two students: Qwen3-8B and Qwen3-4B. We generate ~1.5K trajectories (1.7× SimpleDeepSearcher's 871, sufficient for selection experiments).

**Contribution 1: Mechanistic Diagnostic — Token-level vs Strategy-level Learnability Decoupling**
The core diagnostic motivating the paper. We compute RSR on 500 search-reasoning trajectories and test our mechanistic hypothesis: RSR's token-level rank/surprisal captures whether a student can predict individual tokens but misses whether it can replicate the sequential search strategy. Concretely, we identify trajectories where (a) per-token RSR is favorable (low ratio — tokens are learnable) but (b) search query sequence quality is poor (queries that luck into correct passages without demonstrating transferable search strategy, e.g., copy-pasting entity names vs. formulating decomposed sub-questions). We also identify the reverse: trajectories with high-surprisal query tokens (poor RSR) that demonstrate excellent strategic search (progressive query refinement, evidence triangulation). We measure the Spearman correlation between RSR ranking and actual post-SFT performance on a held-out set for each student. If RSR's Spearman drops significantly from 0.86 (math) on search-reasoning, this validates our decoupling hypothesis and is itself a publishable finding. If RSR maintains high correlation, we report this as a positive generalization result.

**Contribution 2: Search-Decision RSR (SD-RSR) with Weighted Decomposition**
To address the signal dilution problem — search query tokens constitute only ~5-15% of trajectory tokens, so decomposing RSR naively yields negligible difference — we propose SD-RSR with two key design choices:
(a) **Focused computation on search-decision tokens**: Rather than computing RSR over all tokens, we isolate 'search-decision tokens' — query formulation tokens plus search-or-continue decision tokens (where the model decides whether to search again or synthesize). This amplifies the search signal that vanilla RSR dilutes.
(b) **Weighted dual-score**: SD-RSR = (1-α)·RSR_reasoning + α·RSR_search-decision, where α is a tunable weight. We include an ablation over α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} to characterize the search-reasoning tradeoff. The key insight: rather than inventing new metrics, we apply RSR's own token-level informative alignment principle to different functional token classes, weighted by a parameter that captures the student's relative skill gap across search vs reasoning.

**Contribution 3: Cross-Student Trajectory Ranking Divergence**
With two students (Qwen3-4B and Qwen3-8B), we demonstrate that the same trajectory ranks differently for different students — specifically on the search-decision dimension. We hypothesize Qwen3-4B struggles more with reasoning tokens (lower reasoning RSR) while Qwen3-8B's bottleneck is search strategy (lower search-decision RSR), producing divergent optimal α values. This directly validates that student calibration matters for search-reasoning and cannot be captured by student-agnostic approaches.

**Experimental conditions (6 total, each run on both students):**
(1) Random trajectory selection (baseline)
(2) Correctness-only filtering (SimpleDeepSearcher/OpenSeeker baseline)
(3) Vanilla RSR filtering (full-trajectory token-level RSR)
(4) SD-RSR filtering with student-specific optimal α
(5) TopoCurate-style structural filtering adapted to search-reasoning (ablation baseline)
(6) SD-RSR + Gen-SSD adapted student-in-the-loop generation (full method)

**Positioning against related work:**
- vs RSR: We test RSR's generalization claim and propose search-aware decomposition
- vs DASD: DASD addresses distribution mismatch at sequence level; we address trajectory selection at the functional-token level — complementary (one can apply DASD's temperature scheduling during SFT AND our SD-RSR for selection)
- vs T3S: T3S classifies tokens by training dynamics (anchor vs yet-to-learn); we classify by functional role (reasoning vs search-decision) — orthogonal decomposition axes
- vs TopoCurate: TopoCurate selects by trajectory structural quality (student-agnostic); we select by student-trajectory compatibility — included as ablation baseline
- vs Skill-Aware Selection: Similar spirit of targeting student weaknesses, but they define skills by math problem type; we define skill dimensions by functional role in search-reasoning (search skill vs reasoning skill)
- vs OpenSeeker/SimpleDeepSearcher: Their quality filters are student-agnostic; our selection is student-calibrated. Complementary: apply their denoising/curation first, then our student-calibrated selection

**Scale:** ~1.5K trajectories, 6 conditions × 2 students × 2 seeds = 24 SFT runs (but 4B runs are ~40% cheaper). Evaluation on HotpotQA, MuSiQue, 2WikiMHQA dev sets. α ablation: 6 values on best condition = 12 additional runs (small, single-seed).

## Resource Estimate

- **auto_research**: {'claude_api_cost': '$300-500 (~4000-6000 turns)', 'estimated_gpu_cost': '$0 (user-owned hardware)', 'gpu_hours': '~140-170h on 2×RTX PRO 6000 (comprehensive: full α sweep, trajectory count scaling curves, additional Qwen3-1.5B student for 3-point capacity curve)', 'gpu_utilization': 'Very High — 85-95%, both GPUs utilized with pipelined generation → training', 'risk_note': "If SD-RSR's search-decision component shows negligible effect at all α values, fallback position is still strong: the diagnostic (Contribution 1) showing RSR's Spearman drop on search-reasoning + cross-student ranking divergence (Contribution 3) are independently publishable findings.", 'speed_bottleneck': 'Teacher generation is the dominant cost (~30% of total GPU time). Cannot be further parallelized without additional GPUs.', 'team': '0 (after initial setup: Wikipedia index, dataset prep, eval harness)', 'throughput_assumption': 'Same as manual mode', 'timeline': '2-3 weeks with 2×RTX PRO 6000'}
- **human_in_loop**: {'claude_api_cost': '$150-300 (~2000-4000 turns)', 'estimated_gpu_cost': '$0 (user-owned hardware)', 'gpu_hours': '~90-100h on 2×RTX PRO 6000 (agent runs while human reviews diagnostic and α ablation results)', 'gpu_utilization': 'High — 75-85%, agent queues experiments while human reviews', 'risk_note': 'ColBERT index for Wikipedia subset (~5M passages for HotpotQA/MuSiQue) needs ~20GB disk; pre-built indices available from ColBERTv2 releases', 'team': '1 researcher, ~30min/day check-in', 'throughput_assumption': 'Same as manual mode', 'timeline': '3-4 weeks'}
- **manual**: {'estimated_gpu_cost': '$0 (user-owned hardware)', 'gpu_hours': '~85-95h on 2×RTX PRO 6000. Breakdown: Teacher generation (Qwen3-32B AWQ-INT4): 1.5K trajectories × 3K tokens = 4.5M tokens → 25-42h on 1 GPU, ~13-21h on 2 GPUs parallel. SFT Qwen3-8B: 6 conditions × 2 seeds × 2.5h = 30h on 1 GPU. SFT Qwen3-4B: 6 conditions × 2 seeds × 1.5h = 18h on 1 GPU (runs parallel with 8B on second GPU → 30h wall). α ablation: 12 runs × 1.5h avg = 18h (runs after main, single GPU). Evaluation: 24 checkpoints × 3 benchmarks × 0.3h = ~22h (parallel on 2 GPUs → 11h wall). RSR computation: ~3h. Total wall time with 2-GPU parallelism: ~65-75h. Total GPU-hours: ~85-95h.', 'gpu_utilization': 'High — 70-80%, two students enable continuous GPU utilization with pipelined generation → parallel SFT', 'team': '1-2 researchers', 'throughput_assumption': 'Qwen3-32B AWQ-INT4 via vLLM: 30-50 tok/s per RTX PRO 6000', 'timeline': '4-5 weeks'}

## Key References

- Which Reasoning Trajectories Teach Students to Reason Better? RSR Metric (arXiv 2601.14249) — RSR = Σmin(Rank(tₖ),rₘₐₓ)/ΣSurprisal(tₖ), only evaluated on math, authors note extension as future work
- Gen-SSD: Student-in-the-Loop Chain-of-Thought Distillation via Generation-Time Selection (arXiv 2604.02819) — +5.9 over standard KD on math via student-aware trajectory selection
- SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis (EMNLP 2025 Findings) — 871 curated trajectories with four-dimensional student-agnostic quality filtering
- OpenSeeker: Democratizing Frontier Search Agents (arXiv 2603.15594) — 11.7K trajectories with retrospective summarization denoising, student-agnostic
- DASD: Distribution-Aligned Sequence Distillation (arXiv 2601.09088) — divergence-aware sampling + temperature scheduling for teacher-student distribution alignment, SOTA with 448K samples
- T3S: Training-Trajectory-Aware Token Selection (arXiv 2601.10348) — token-level bifurcation into imitation-anchor vs yet-to-learn tokens, reconstructing training objectives
- TopoCurate: Modeling Interaction Topology for Tool-Use Agent Training (arXiv 2603.01714) — topology-based dual-selection for agent SFT trajectories, student-agnostic structural quality
- OpenResearcher: Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis (arXiv 2603.20278) — 97K+ search trajectories, SFT alone unlocks significant gains
- What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces? (arXiv 2602.01611) — semantic learning vs interface shortcutting, PIPE evaluation protocol
- Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation (arXiv 2601.10109) — student skill-weakness-aware selection, +1.6% with only 1K examples
- AceSearcher: Bootstrapping Reasoning and Search via Reinforced Self-Play (NeurIPS 2025 Spotlight)
- HybridDeepSearcher: Hybrid Parallel-Sequential Reasoning for Web Search (ICLR 2026)
- Search-R1: Training LLMs to Reason and Leverage Search Engines with RL (COLM 2025)
- Why Does Self-Distillation Sometimes Degrade Reasoning? (arXiv 2603.24472)
- A Survey of On-Policy Distillation for Large Language Models (arXiv 2604.00626)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (Technical Report 2025)

## Reviewer Evaluation

**Scores**:
  - clarity: **{'rationale': 'Excellent positioning against 6+ related works with clear differentiation. The three contributions are well-defined and independently valuable. The experimental design (6 conditions × 2 students × 2 seeds) is thorough, and the α ablation directly addresses the signal dilution concern raised in prior rounds.', 'score': 4}**/5
  - feasibility: **{'rationale': "Resource estimates are now realistic: 30-50 tok/s for Qwen3-32B AWQ-INT4 is confirmed plausible on RTX PRO 6000; 85-95h total GPU time is tight but within the <100h constraint; local BM25+ColBERT retrieval eliminates API dependency; 1.5K trajectories is well-calibrated (1.7× SimpleDeepSearcher's 871). EMNLP ARR May 25 timeline (47 days) is tight but achievable for a focused empirical study.", 'score': 4}**/5
  - impact: **{'rationale': 'Search-reasoning distillation is a hot area with 6+ papers at top venues (NeurIPS, EMNLP, ICLR, COLM) in 2025-2026. A principled approach to student-calibrated trajectory selection for this domain fills a clear need. The mechanistic insight about token-level vs strategy-level decoupling could generalize to broader agent trajectory distillation settings.', 'score': 4}**/5
  - novelty: **{'rationale': 'The gap is genuine and independently verified — no existing work combines student-calibrated trajectory selection with search-reasoning distillation. The mechanistic hypothesis (token-level vs strategy-level learnability decoupling) is well-articulated and provides a testable, publishable insight beyond a simple domain transfer of RSR.', 'score': 4}**/5
