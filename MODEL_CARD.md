---
language:
- en
license: apache-2.0
tags:
- reasoning
- mathematical-reasoning
- reinforcement-learning
- grpo
- rlvr
- self-consistency
- consensus
- self-correction
- lora
- qwen
base_model: Qwen/Qwen2.5-1.5B-Instruct
---

# Model Card: Self-Correcting Micro-Reasoner (Qwen2.5-1.5B-Instruct-GRPO)

## Model Overview
- **Model Name:** Self-Correcting Micro-Reasoner
- **Base Architecture:** Qwen2.5-1.5B-Instruct (1.54B parameters)
- **Training Strategy:** Three-phase post-training:
  1. **Stage B:** Supervised Fine-Tuning (SFT) cold-start (250 steps) for structured `<think>...</think><answer>...</answer>` reasoning format.
  2. **Stage D1:** Group Relative Policy Optimization (GRPO) (100 steps, $G=4$) with deterministic, verifiable mathematical rewards (RLVR).
  3. **Stage D2:** Scaled GRPO (150 steps, $G=8$ rollouts, $T=0.8$) to expand exploration and minimize zero-variance gradient dropout.
- **Inference Scaling:** Test-time compute via Self-Consistency Consensus Voting ($K=8$, $T=0.7$, $\text{top\_p}=0.9$).
- **Target Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, Blackwell sm_120).
- **License:** Apache 2.0

---

## Intended Use and Scope
- **Primary Use:** Solving grade-school, multi-step algebraic, geometric, motion/rate, and structured arithmetic problems with explicit step-by-step reasoning and parseable answers.
- **Secondary Use:** Benchmarking reinforcement learning from verifiable rewards without learned critics on consumer GPUs.
- **Out-of-Scope Uses:** 
  - General conversational chat where reasoning tags are undesirable.
  - High-stakes financial, medical, or legal calculations without external deterministic tool verification.
  - Non-mathematical factual question-answering.

---

## Measured Performance Progression (Held-Out Test Set, N=100)

| Setting | Exact Accuracy | Parse Success Rate | Invalid Output Rate | Mean Output Tokens |
|---|---|---|---|---|
| **Stage A: Unmodified Base (Greedy)** | 28.00% | 38.00% | 62.00% | 389.4 |
| **Stage B: Cold-Start SFT (Greedy)** | 54.00% | 100.00% | 0.00% | 131.2 |
| **Stage D1: SFT + GRPO (G=4, Greedy)** | 58.00% | 100.00% | 0.00% | 138.7 |
| **Stage D2: Scaled GRPO (G=8, Greedy)** | 57.00% | 100.00% | 0.00% | 138.0 |
| **Stage D1 + Consensus@8 Majority Vote** | **65.00%** | 100.00% | 0.00% | 1075.5 (8 rollouts) |
| **Stage D2 + Consensus@8 Majority Vote** | **67.00%** | 100.00% | 0.00% | 1082.0 (8 rollouts) |
| **Stage D2 Pass@8 (Oracle Upper Bound)** | **81.00%** | 100.00% | 0.00% | 1082.0 (8 rollouts) |

---

## Self-Correction Performance
Evaluated under a strict two-pass revision protocol (Exp C on 50 held-out samples):
1. **Initial Pass Accuracy:** 54.00% (27/50)
2. **Revised Pass Accuracy:** 56.00% (28/50)
3. **Correction Rate (Initial False $\to$ Revised True):** 26.09% (6 of 23 initial errors successfully corrected).
4. **Regression Rate (Initial True $\to$ Revised False):** 18.52% (5 of 27 initial correct answers mistakenly altered).
5. **Net Accuracy Change:** +2.00% absolute.
6. **Token Overhead:** $2.21\times$ the single-pass token count (265.8 total tokens vs 120.2 tokens).
7. **Prompted Verification (Exp B):** 6.00% accuracy — simple prompting to verify calculation without revision structure caused catastrophic overthinking.

---

## Key Research Conclusions
1. **Format Alignment is Foundational:** The base model's apparent failure (28%) is primarily an output formatting issue. SFT format tuning fixes 100% of parse errors and boosts accuracy to 54%.
2. **RLVR Refines Reasoning Without Bloat:** GRPO increases greedy accuracy to 58.00% while condensing P95 token length from 226.0 to 201.0 tokens, showing zero evidence of length hacking.
3. **Test-Time Compute Bridges the Gap to 67%–81%:** Majority voting across 8 sampled rollouts pushes accuracy from **58.00% to 67.00%**. At least one correct reasoning trajectory exists in **81.00% of problems (Pass@8)**, confirming that the 1.5B model's latent capacity is high but subject to occasional arithmetic slips.
