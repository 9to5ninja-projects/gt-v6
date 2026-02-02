# Training hybrid SSM-attention models: what works and what fails

The most reliable path to training hybrid state space + sliding window attention architectures combines **joint end-to-end training with 1:7 attention-to-SSM layer ratios**, sequence length curriculum from 256→4096 tokens, and careful numerical precision (FP32 for SSM parameters). Empirical evidence from Jamba, Gated DeltaNet, and Based demonstrates that attention layers handle all retrieval while SSMs provide efficient long-range context—a complete functional segregation confirmed by ablations showing **0% retrieval accuracy when attention is removed** from hybrids. The critical surprise from AI21's experiments: Mamba-1 outperforms Mamba-2 in hybrid architectures, likely because attention already provides the precise recall that Mamba-2's larger state was designed to address.

## Curriculum training works but requires narrow hyperparameter windows

Sequence length progression delivers measurable benefits for SSM training. DeepSpeed curriculum learning achieves **3.3× faster GPT-2 pretraining** by starting at shorter sequences (min_difficulty=64-256 tokens) and progressively increasing. The "Grow-P2" curriculum—8 cycles from 256 to 8192 tokens using powers-of-2 progression—outperforms both constant-length training and linear progression schedules. For Mamba specifically, DNA pretraining used warmup from 2^10 to 2^20 tokens, enabling extrapolation to **1M tokens (4000× training length)**—no other architecture exceeds 2× extrapolation.

The critical caveat: SSMs operate in an **extremely narrow learning rate window** compared to Transformers. Over 3,000 training runs (~20,000 GPU hours) characterized this sensitivity, revealing that prior benchmark comparisons likely underestimated SSM capability due to suboptimal LR selection. Concrete configurations that work include Gated DeltaNet trained at 4K tokens with 0.5M tokens per batch, achieving 45K tokens/second on H100s, and RWKV-14B trained at fixed 1024 context length requiring ~140,000 A100 hours.

| Model | Sequence Length | Batch Size | Training Tokens |
|-------|----------------|------------|-----------------|
| Mamba-3B | 2048 | - | 300B |
| Gated DeltaNet 1.3B | 4096 | 0.5M tokens | 100B |
| RWKV-14B | 1024 | 128-256 seqs | 300B |
| Jamba-52B | Variable | - | 250B+ |

Failed curriculum approaches include too-fast progression (causes severe overfitting and loss divergence), sequential easy→hard without mixing (loses overall accuracy despite gains on hard concepts), and mismatched LR schedules (token-based decay required, not sample-based).

## Joint training dominates; staged component pretraining shows no benefit

All production hybrid models—Jamba, Zamba, Bamba, Nemotron-H, RecurrentGemma—use **joint end-to-end pretraining** rather than separate SSM/attention pretraining. No published evidence supports freezing one component while training the other. The "staged training" that does help is pretraining + annealing phases: Zamba trained ~950B tokens in phase 1, then 50B high-quality tokens for annealing. Zyphra's cookbook recommends annealing data at **10-15% of total pretraining budget** with 50-70% replay from original distribution.

Component-specific learning rates appear unnecessary—all major hybrids use unified LR schedules. Concrete values: Bamba-9B uses peak LR **3e-4** with cosine decay to 1e-5; Zamba-7B uses 1.5e-4→7.5e-5; Nemotron-Nano-12B uses WSD schedule with 4.5e-4 stable LR. Standard warmup (2000-4000 steps) works for both components. No auxiliary memory losses appear in production—standard next-token prediction dominates, with data quality (not loss function changes) driving memory improvement.

The critical stability requirement: **keep SSM parameters in FP32**. From the official Mamba repository: "SSMs are sensitive to their recurrent dynamics. If you are experiencing instabilities, as a first step please try a framework storing parameters in fp32." BF16 overflow and DeepSpeed incompatibilities are documented across multiple practitioners. Nemotron-H achieved first successful FP8 hybrid training using per-tensor scaling.

## Attention-to-SSM ratios converge on 1:4 to 1:7 across architectures

Jamba's ablations at 1.3B parameters and 250B tokens tested pure attention, pure Mamba, 1:3, and 1:7 ratios. The finding: **"There is no substantial difference between 1:3 and 1:7 ratios"** in quality (identical OLLM scores of 37.2), so 1:7 was chosen for compute efficiency. Pure Mamba catastrophically fails in-context learning: IMDB accuracy drops from 90.9% (hybrid) to 48.8% (pure Mamba). The hybrid with only 1 attention layer per 8 total exhibits ICL "similar to vanilla Transformers."

| Architecture | Attention:SSM Ratio | Empirical Justification |
|--------------|---------------------|------------------------|
| Jamba | 1:7 | "No difference from 1:3, more efficient" |
| Zamba | 1:6 | Shared attention every 6 Mamba blocks |
| Qwen3-Next | 1:3 (25% attention) | Gated DeltaNet + GQA every 4th layer |
| Griffin | 1:2 | 2 RG-LRU blocks per 1 local attention |
| H3 | Minimal (2 layers) | Second and middle layers only |

The "SSM prepares, attention retrieves" hypothesis has strong empirical support. A study across RecurrentGemma and Jamba-Mini found **complete functional segregation**: retrieval depends exclusively on self-attention, with SSM layers showing zero compensatory mechanisms even with improved prompting. Critically, **sparsifying attention to just 15% of heads maintains near-perfect retrieval** while preserving 84% MMLU performance—the attention budget can be dramatically reduced without losing the retrieval function that defines its contribution.

## State sizing follows hardware constraints more than theoretical optimums

Mamba-1 used state expansion factor N=16 (hardware-limited), while Mamba-2 achieved N=64-128 through the SSD algorithm leveraging tensor cores. The relationship between state size and model dimension appears decoupled in practice: Codestral 7B uses state_size=128 with hidden_size=4096, giving ratio ~0.03. Memory per SSM layer follows: D × N × 2 bytes (FP16).

The **minimum state capacity for associative recall** depends on task requirements. Gated convolution architectures require state dimension scaling linearly with sequence length to solve MQAR, while attention solves it with constant dimension (d≥32 sufficient). For real-world retrieval, Jamba's 1:7 hybrid with modest SSM state achieves competitive results because attention handles precise recall. The scaling insight from AI21: **Mamba-1 + Attention outperforms Mamba-2 + Attention** in hybrids, suggesting "advantages of Mamba-2 (larger state size) are less significant when attention layers are interleaved, as they can pool information from entire context."

KV cache memory savings are dramatic: at 256K context, Llama-2-70B requires 128GB while Jamba (12B active) requires only **4GB**. Inference throughput improvements of 2.5-5× over pure Transformers are consistent across Bamba, Jamba, and Nemotron-H.

## Short convolutions and delta rules provide largest associative recall gains

The **Convolution-Augmented Transformer (CAT)** technique places short 1D convolutions (kernel size 3-4) before Q/K/V projections, enabling single-layer Transformers to achieve **100% MQAR accuracy** where standard single-layer models reach only 60-80%. The convolution creates bigram-like key-value pairs by shifting embeddings. This transfers to other architectures, though removing convolution from Mamba specifically doesn't hurt—S6 provides sufficient expressivity.

Taylor feature maps for linear attention approximation work because they preserve the "spikiness" of softmax. Second-order Taylor expansion (exp(x) ≈ 1 + x + x²/2) achieves the best recall-throughput tradeoff, with Based's custom CUDA kernels delivering **24× higher throughput than FlashAttention-2** at 1.3B parameters.

**Gated DeltaNet** represents the current state-of-the-art for SSM recall (ICLR 2025). It combines:
- **Gating (α_t)**: data-dependent rapid memory erasure
- **Delta rule (β_t)**: targeted key-value replacement that minimizes MSE

| Model | Wiki PPL | Zero-shot Avg | S-NIAH-1 (8K) |
|-------|----------|---------------|---------------|
| Mamba2 | 16.56 | 54.89 | 30.4 |
| DeltaNet | 17.71 | 52.14 | 98.8 |
| Gated DeltaNet | **16.42** | **55.32** | 91.8 |

DeltaNet excels at retention but lacks memory-clearing; Mamba2's gating clears uniformly but hurts retention. Gated DeltaNet combines both advantages.

## Documented failures reveal fundamental SSM limitations

**Pure SSM for in-context learning fails completely.** Jamba ablations showed pure Mamba "struggles to develop ICL capabilities"—it produces "Very Good" instead of "Positive" because it cannot follow input-output format from examples. A single attention layer (1/8 of total) restores ICL to Transformer-level performance.

**Mamba-2 underperforms Mamba-1 in hybrids.** This counterintuitive result from AI21 suggests Mamba-2's restricted expressivity (scalar-valued A) hurts when combined with attention, which already handles the precise recall Mamba-2's larger state was designed for. Recommendation: use Mamba-1 blocks in hybrid architectures.

**Architectures that failed at scale** include StripedHyena (lagged behind Mistral-7B despite novel design), H3 (improvements didn't transfer beyond small scale), and pure SSM scaling (before Jamba, no successful hybrid beyond 3B parameters). Training instabilities specific to memory components include vanishing/exploding gradients in deep Mamba stacks (mitigated by RMSNorm after each sublayer), state saturation in linear attention (can only add associations, not erase—"the enemy of memory is not time; it's other memories"), and numerical precision degradation (mixed-precision fine-tuning causes Mamba performance drops that don't occur in Transformers).

**Fundamental limitation proven**: "Any language model with fixed-size memory (any SSM) fails to copy long random strings." Transformers copy exponential-length strings; SSMs are limited by fixed state dimension. Phonebook retrieval shows Transformers "substantially outperform Mamba across model sizes."

## Practical implementation recommendations

**Training configuration:**
- Learning rate: 1.5e-4 to 4.5e-4 peak, cosine or WSD schedule
- Warmup: 2000-4000 steps, linear or quadratic
- Batch size: 0.5-1.5M tokens per batch
- Sequence length: start 256-512, increase to 4096-8192
- Optimizer: AdamW (β1=0.9, β2=0.95), weight decay 0.1
- Precision: FP32 for SSM parameters, AMP for rest

**Architecture choices:**
- Attention-to-SSM ratio: 1:5 to 1:7 for efficiency, 1:3 to 1:4 for maximum recall
- State expansion: N=64-128 (Mamba-2 style)
- Use Mamba-1 blocks rather than Mamba-2 in hybrids
- RMSNorm essential for training stability at scale
- Place attention blocks at regular intervals, not clustered

**What to avoid:**
- Pure SSM for tasks requiring ICL or exact retrieval
- BF16/FP16 for SSM state tracking without careful validation
- Quantizing Mamba blocks (AI21 explicitly recommends exclusion)
- Expecting SSM layers to contribute to retrieval—they don't

## Conclusion

The empirical evidence points to a clear functional division: **attention provides eidetic (verbatim) memory over finite context, while SSMs provide compressed long-term memory that fades with distance**. The optimal hybrid uses sparse attention (15-25% of layers) for retrieval and dense SSM layers for efficient context processing. Curriculum training helps but requires careful LR tuning; joint training dominates staged approaches; and Gated DeltaNet's combination of gating + delta rule represents the current frontier for SSM-based recall. The most surprising finding—Mamba-1 outperforming Mamba-2 in hybrids—suggests the field may have over-indexed on SSM state capacity when the real bottleneck is the attention-SSM interaction pattern.