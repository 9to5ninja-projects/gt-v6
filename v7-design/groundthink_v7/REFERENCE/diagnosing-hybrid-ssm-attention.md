# Diagnosing hybrid SSM-attention training: evidence-based validation

**Your +8% accuracy differential is likely insufficient to confirm genuine memory usage** without additional statistical testing and ablation protocols. The research reveals that simple synthetic tasks like needle-in-haystack are poor predictors of real retrieval capability—models achieving 95%+ NIAH accuracy often fail catastrophically on harder benchmarks. Hybrid SSM-attention architectures face a fundamental challenge: gated convolutions can solve synthetic associative recall perfectly while a **70M attention model outperforms a 1.4B Hyena model on real-world retrieval**. This gap, documented by Stanford's Zoology research, suggests your architecture choices and validation methodology matter far more than raw synthetic accuracy.

---

## Auxiliary loss formulations that actually work for state utilization

The most concrete implementation for forcing state utilization comes from **Trinh et al. 2018**, which used reconstruction-based auxiliary losses for RNN long-term dependencies. The approach samples random anchor points in sequences and reconstructs subsequences occurring *before* the anchor, forcing the network to store temporal information in hidden state. The exact formulation uses L2 distance minimization with scheduled sampling for the decoder, BPTT truncated to 300 time steps, and subsequence lengths of 600 tokens.

For loss weighting (α values), the empirical evidence converges on **α = 0.1–0.3 for related auxiliary tasks**. A systematic study by Vivien (2019) found that when auxiliary tasks are similar to the primary task, λ ≤ 1 helps; when tasks are dissimilar, even λ ≥ 0.3 causes measurable harm. The MoE load balancing literature suggests α ≈ 10⁻³ for expert balancing, while spiking neural network research found α > 0.3 yielded better results for classification auxiliaries.

**Documented cases where auxiliary losses hurt performance:**

- Input reconstruction in RL (UNREAL study): Agents learned faster initially but final performance was *worse* than vanilla A3C—the network struggled reconstructing irrelevant features
- Abstract auxiliary concepts (DVS-CIFAR10): Using "living vs non-living" as auxiliary categorization caused performance decline because the concept was too difficult to leverage
- Gradient conflicts (Du et al. 2018): When ∇L_auxiliary and ∇L_primary form an obtuse angle, auxiliary gradients cancel primary gradients, potentially halting learning

**Reconstruction losses outperform contrastive losses for sequence models** by roughly 14–15% versus 7–8% improvement in time series applications (PatchFormer 2024). This makes sense theoretically: reconstruction forces dense information storage across the sequence, while contrastive learning primarily improves discriminative representations. For SSMs specifically, no documented auxiliary losses appear in the original Mamba/S4 papers—the architecture is "sensitive to recurrent dynamics," and auxiliary losses may interfere with learned HiPPO matrix dynamics.

**Scheduling recommendation:** Start constant, then adapt. Trinh et al. used constant weights successfully throughout training. However, if you observe gradient conflicts (cosine similarity between gradients < 0), implement the projection method: project auxiliary gradients onto the half-space with positive similarity to primary gradients.

---

## What legitimate loss curves look like at your scale

For **10M–100M parameter** language models trained on sufficient data, Chinchilla scaling laws predict:

| Model size | Optimal tokens | Expected final loss | Perplexity range |
|------------|----------------|---------------------|------------------|
| 10M params | ~200M tokens | 4.0–4.5 nats | 50–90 |
| 50M params | ~1B tokens | 3.5–4.0 nats | 33–55 |
| 100M params | ~2B tokens | 3.2–3.7 nats | 25–40 |

**Training dynamics follow a predictable pattern**: models first learn unigram distributions (token frequencies) achieving r > 0.95 correlation across runs within the first ~1K steps, then gradually learn n-gram patterns with decreasing cross-run similarity reaching a local minimum around 5-gram learning, and finally refine tail predictions with smoothly decreasing loss.

**Red flags indicating false convergence:**

- **Loss drops too fast**: Achieving final loss in <10% of expected training steps
- **Loss suspiciously low**: Cross-entropy below 2.5 for a 100M model almost certainly indicates memorization or data leakage
- **Train/val gap near zero**: Should be small but present; identical values suggest degenerate solutions or data contamination
- **Spike-free training**: Some variability is normal; perfectly smooth curves warrant investigation

**To detect degenerate solutions**, measure repetition rate in generated text. Human baseline shows ~0.28% repetition in the first 200 tokens; beam search produces 73.66% (severely degenerate). Compute distinct n-gram ratios—values below 0.5 for bigrams indicate repetition collapse. Critically, check the perplexity of your model's *outputs*: degenerate models produce text with unnaturally low perplexity (~1.5), while healthy models produce text around human-like perplexity (12–20 for WebText-quality text).

**Validation metrics resistant to gaming**: Perplexity on held-out domains (train on 2020 news, evaluate on 2015), LAMBADA for long-range understanding, and generation-based evaluation across temperature ranges. A healthy model remains coherent at low temperature and diverse at high temperature; degenerate models are repetitive at low temperature and incoherent at high.

---

## vLLM supports hybrid architectures—but constraints exist

**Yes, vLLM V1 (November 2025) treats hybrid models as first-class citizens.** Supported architectures include Jamba, Mamba, Mamba-2, Falcon Mamba, Bamba, Zamba2, and IBM's Granite 4.0-H (9:1 Mamba-2/Transformer ratio). The V0 implementation was "hacky" with SSM state allocated separately from KV cache, but V1 provides unified memory allocation managing both.

**Key architectural constraints vLLM imposes:**

- **Memory allocation differences**: Attention KV cache grows linearly (~64 KiB per 16-token block) while Mamba state is fixed (~2.57 MiB per sequence regardless of length)—at 128K tokens, Mamba uses ~200x less memory than attention
- **Page alignment requirements**: Attention block sizes are automatically increased until they align with Mamba's page size, sometimes creating unusual block sizes like 672 tokens per attention block
- **Feature limitations**: Prefix caching for Mamba-2 is experimental; KV cache offloading and prefill-decode disaggregation are incompatible with SSM state; chunked prefill has "major performance issues"

**Scale considerations:** vLLM integration matters minimally below 1B parameters—native PyTorch suffices for research. At 7B+ or in production deployment (any size), paged attention, continuous batching, and memory management become critical. For long context (>32K tokens), hybrid models provide substantial memory advantages making vLLM integration worthwhile. Benchmark results show vLLM V1 delivers **2–91% throughput improvements** over V0 for hybrid models, with the largest gains on MoE hybrids.

**This is largely a "solve later" problem** for research at 10M–100M scale. However, IBM's approach to Granite 4.0 suggests major organizations treat inference compatibility as a design constraint: they explicitly collaborated with vLLM and llama.cpp teams *before* release. If you're designing novel attention variants or custom SSM formulations intended for production, prototype vLLM integration early.

---

## Curriculum schedules that work: specific numbers

The most concrete guidance comes from **DeepSpeed's curriculum learning research** for GPT-2/GPT-3 scale models:

| Parameter | Recommended value |
|-----------|-------------------|
| Curriculum duration | 15,000–60,000 steps (larger models need longer) |
| Starting sequence length | 8 tokens (small models) or 64 tokens (billion-scale) |
| Target sequence length | Full training length (1K–2K tokens) |
| Difficulty step size | 8 (FP16) or 16 (INT8) for Tensor Core efficiency |

This achieved **3.3x faster** GPT-2 pre-training and enabled training with 8x larger batch sizes and 4x larger learning rates without divergence. The key insight: training instability correlates strongly with early long sequence lengths—most instabilities occur in the first 5K steps when using full sequence length.

**Retrieval-first versus LM-first versus interleaved:** The evidence favors **combined/interleaved approaches**. The "Learning to Execute" paper (Zaremba & Sutskever 2014) found that combined curriculum (mixing naive progression with random sampling from full difficulty space) outperformed both naive curriculum and random sampling alone. For the DNC, curriculum learning started with simpler graphs and gradually increased complexity; the LSTM baseline couldn't complete even the easiest curriculum task.

**Retrieval warmup duration:** In-context learning research suggests curriculum phases comprising **5% of total training**—context length expanded from 1 to 41 over the first 30K of 600K total steps. Cerebras's Variable Sequence Length training uses a two-stage approach: 75% of training at 512 tokens, 25% at full context (2048–8192), achieving **29% fewer FLOPs** with equivalent downstream performance.

**Impact on final perplexity:** Curriculum training generally helps both convergence speed and final performance. DeepSpeed reports better token-wise convergence and better zero-shot results on WikiText-103/LAMBADA. However, risks exist: staying at one difficulty level too long causes training divergence when switching; perplexity-based ordering shows strong early gains but drops in later phases. Monitor validation perplexity and use binary search to find optimal curriculum duration.

---

## Your +8% differential requires statistical validation

**+8% accuracy improvement between state-present and state-zeroed conditions is potentially meaningful but requires statistical testing** given typical sample sizes. The guidelines from ML statistics literature:

| Differential | Interpretation |
|--------------|----------------|
| <5% | Likely noise/marginal |
| 5–10% | Potentially meaningful, requires testing |
| 10–20% | Likely meaningful |
| >20% | Almost certainly meaningful |

**Required statistical protocol:**

1. Use **McNemar's test** for paired nominal data (recommended for comparing two conditions on the same test set)
2. Alternatively, **5×2 cross-validation with modified t-test** (Dietterich 1998), which accounts for variance across train/test splits
3. Require **n ≥ 500 test samples** per condition for 80% power to detect 8% differences
4. Report effect sizes (Cohen's d > 0.5 for medium effect) alongside p-values

**The gold-standard ablation protocol** is state zeroing: run inference normally (Accuracy_full), then reset hidden state to zeros at test time (Accuracy_zeroed). Compute Memory Reliance Score = (Accuracy_full - Accuracy_zeroed) / Accuracy_full × 100%. **A score above 70% strongly indicates genuine state-based retrieval.**

**Additional validation layers:**

- **Permutation tests**: Shuffle input order on position-independent tasks; genuine content-based retrieval should maintain performance
- **Out-of-distribution position tests**: Test on sequences where targets appear at positions never seen during training; genuine memory maintains >80% of training accuracy
- **Information-theoretic threshold**: Mutual information between state and target should exceed **1.0 bits** (measured via MINE or similar estimators)
- **Linear probe accuracy**: If a linear classifier trained on frozen hidden states achieves >70% of full model accuracy on retrieval targets, information is clearly encoded

**Red flags for shortcut learning**: Performance depends heavily on input position, accuracy varies >15% across data slices, model shows >20% accuracy drop on out-of-distribution positions, or attention/state patterns focus on positions rather than content.

---

## Synthetic task performance poorly predicts real-world retrieval

The most important finding for your diagnostic concerns: **simple synthetic tasks are unreliable predictors of real retrieval capability**, especially for non-Transformer architectures. The Zoology research from Stanford's Hazy Lab documented that gated convolutions (Hyena, H3, RWKV) solved synthetic associative recall perfectly—yet a **70M parameter attention model outperformed a 1.4B Hyena model on real-world associative recall** (2.41 vs 3.43 perplexity).

**Which synthetic tasks actually predict real performance:**

| Task | Predictiveness | Reason |
|------|----------------|--------|
| Standard Needle-in-Haystack | ❌ Low | Too easy; solved by position tracking |
| Single-query associative recall | ❌ Low | Doesn't capture multi-query nature of real language |
| Simple copy task | ❌ Low | Can be solved by convolution kernels without content reasoning |
| **Multi-Query Associative Recall (MQAR)** | ✅ High | Multi-query nature matches real language demands |
| **Induction heads** | ✅ High | Direct mechanism for in-context learning |

**The transfer gap is substantial.** Models achieving >99% on standard NIAH fail to exceed Llama2-7B baseline (85.6%) on harder RULER benchmark tasks at their claimed context lengths. The ICLR 2025 paper "From Artificial Needles to Real Haystacks" found that finetuning on synthetic key-value retrieval improved real-world multi-document QA by only **10.5%**—meaningful but far from solving the problem.

**Minimum model sizes for observable transfer:**

| Architecture | Minimum effective size |
|--------------|------------------------|
| Transformers | ~70M parameters |
| Pure Mamba | 2.8B+ (still lags on recall-intensive tasks) |
| RWKV | 1.5B+ for meaningful transfer |
| Mamba-2-Hybrid | 8B needed to match Transformers |

**For your 10M–100M parameter hybrid**: Expect significant gaps between synthetic and real performance. Focus on MQAR rather than simple NIAH. If synthetic MQAR accuracy exceeds 90% but real-world retrieval-intensive tasks show <70% accuracy, this pattern is typical and doesn't necessarily indicate bugs—it reflects architectural limitations that persist even at larger scales without attention components.

---

## Practical diagnostic checklist

**Before trusting your current results, verify:**

1. ☐ Loss values match scaling law predictions (3.2–4.5 for 10M–100M range)
2. ☐ Generated samples show diversity (Self-BLEU ~0.30–0.35, not >0.40)
3. ☐ State zeroing causes >50% accuracy drop on memory tasks (ideally >70%)
4. ☐ Accuracy differential is statistically significant (p < 0.01, n > 500)
5. ☐ Out-of-distribution positions maintain >80% of training accuracy
6. ☐ Mutual information between state and retrieval target exceeds 1.0 bits
7. ☐ Perplexity of generated text matches human-like range (12–20)
8. ☐ Performance on MQAR predicts performance on retrieval-intensive natural language tasks

If multiple checks fail, your results likely reflect false validation rather than genuine progress. The combination of all checks passing provides reasonable confidence that the model has developed authentic retrieval capabilities through its state mechanism rather than exploiting positional or statistical shortcuts.