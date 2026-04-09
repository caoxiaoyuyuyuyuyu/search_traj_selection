# Search Trajectory Selection — Research Roadmap

> 最后更新: 2026-04-09 10:30 UTC | 更新原因: 根据 Reviewer 反馈补充 ground truth 定义、search-decision token 识别、评估指标、时间线 gates

## 研究目标
验证 student-calibrated trajectory selection 是否从数学 CoT 蒸馏迁移到 search-reasoning 蒸馏。核心假设：RSR 的 token-level rank/surprisal 无法捕捉 search-reasoning 中的 strategy-level 可学性。提出 SD-RSR 通过 search-decision token 加权分解解决。

**目标会议**: EMNLP 2026 ARR (May 25) | **降级方案**: Findings track 或 workshop

## 三个贡献
1. **Mechanistic Diagnostic**: RSR 在 search-reasoning 上 Spearman 是否显著低于数学 (0.86)
2. **SD-RSR**: Search-Decision RSR，聚焦 search-decision tokens + α 加权双分数
3. **Cross-Student Divergence**: 同一 trajectory 对 4B/8B 学生排名不同

## 关键方法论定义

### Ground Truth (Contribution 1)
与 RSR 原论文一致：ground truth = post-SFT 实际性能。
- 将 trajectories 分为 K 个子集，每个子集训练学生，评估 EM/F1
- RSR 对每个子集打分 → Spearman(RSR ranking, actual EM ranking)
- 对两个学生分别计算，对比数学 CoT 上的 0.86 基线

### Search-Decision Token 识别 (Contribution 2)
Pattern-based 解析（不训练分类器），依赖 trajectory 结构化格式：
- **Query tokens**: `<search>`/`<query>` 标签内的 token
- **Decision tokens**: think→search/synthesize 决策点（"I need to search for..."/"Based on the results..."）
- 具体 pattern 在 trajectory 生成后根据实际格式确定
- 鲁棒性验证：随机扰动 pattern boundary ±5 tokens，观察 SD-RSR 排名稳定性

### 评估指标
- 主指标: EM (Exact Match), F1 (token-level)
- 数据集: HotpotQA, MuSiQue, 2WikiMHQA dev sets (held-out)
- 每个 condition 报告 mean ± std (2 seeds)

## 实验条件 (6 conditions × 2 students × 2 seeds)
1. Random selection (baseline)
2. Correctness-only filtering (SimpleDeepSearcher/OpenSeeker baseline)
3. Vanilla RSR (full-trajectory token-level)
4. SD-RSR with student-specific optimal α
5. TopoCurate-style structural filtering (adapted)
6. SD-RSR + Gen-SSD student-in-the-loop (full method)

## 时间线与 Go/No-Go Gates

| Phase | 内容 | Deadline | Gate |
|-------|------|----------|------|
| 0 | 基础设施 + 数据 + 评估 pipeline | **4/15** | 未完成 → 转 findings/workshop |
| 1 | Teacher trajectory 生成 (1.5K) | 4/22 | <500 有效 trajectory → 调整生成策略 |
| 2 | RSR diagnostic + SD-RSR 实现 | 4/28 | RSR Spearman ≥0.8 → 调整 story |
| 3 | SFT 训练 (24 runs) + α ablation | 5/10 | — |
| 4 | 评估 + 分析 + 写作 | 5/22 | 5/18 前结果不完整 → 转 findings |

## 当前阶段: Phase 0 — 基础设施搭建

## 短期任务
- [ ] 项目代码脚手架搭建
- [ ] 数据集下载与预处理（HotpotQA, MuSiQue, 2WikiMHQA）
- [ ] Wikipedia 子集准备 + BM25 索引构建
- [ ] 评估 pipeline 搭建（EM/F1 on multi-hop QA）
- [ ] vLLM + Qwen3-32B AWQ-INT4 部署验证
- [ ] RSR 基线实现
- [ ] 远程代码路径配置

## 基础设施
- Server: autodl-svg-retrieval (connect.westd.seetacloud.com), 2 GPU
- Git: github.com/caoxiaoyuyuyuyuyu/search_traj_selection
- W&B: xiaoyucorynne-/search_traj_selection

## 风险与缓解
- RSR 仍高相关 → story 调整为 "RSR generalizes + SD-RSR further improves"
- SD-RSR 不显著 → Contribution 1 + 3 独立发表
- 1.5K trajectories 不够 → 扩展到 2K（余量充足）

## 进展摘要
- [4/9] 项目初始化，ROADMAP 创建，Agent 开始代码脚手架

## 关键决策日志

| ID | 日期 | 摘要 | 关联实验 |
|----|------|------|----------|
| D001 | 2026-04-09 | 采纳 Reviewer 反馈：补充 ground truth 定义、search-decision  | — |

