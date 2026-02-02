# Honest Test Suite for GroundThink v7

## The Problem With Previous "Validation"

| What We Tested | Why It Was Bullshit |
|----------------|---------------------|
| MARKER/CUE tokens | Model learns "when you see CUE, output the thing after MARKER" — that's pattern matching, not memory retrieval |
| Random needle IDs | No semantic content, no interference from similar values |
| Fixed vocab 50257 | Never tested on actual language distributions |
| seq_len=128 | Goldfish memory — real retrieval needs 1K+ |
| Single needle | Real recall needs multiple competing memories |
| Synthetic data | Zero transfer to actual language modeling |

---

## Controlled Tests That Can't Lie

### Test 1: Factual Recall Without Magic Tokens

Use real text with naturally occurring facts:

```
"The capital of France is Paris. The largest planet is Jupiter. 
[500 tokens of distractor text]
Question: What is the capital of France?"
```

**Pass condition:** Model outputs "Paris" with higher probability than random baseline.

**Why it's hard to fake:** No MARKER token to trigger storage. Model must learn what's worth remembering.

---

### Test 2: Entity Tracking (Competing Memories)

```
"Alice has a red ball. Bob has a blue car. Carol has a green hat.
[variable distractor length]
What does Bob have?"
```

Scale to 10, 20, 50 entities. Measure:
- Accuracy vs. number of entities
- Accuracy vs. distance from fact to query
- Confusion matrix (does it retrieve wrong entity's attribute?)

---

### Test 3: Perplexity on Copy-Dependent Tokens

From real corpus (WikiText, OpenWebText), identify tokens that require long-range copy:
- Repeated names
- Pronouns with distant antecedents
- Technical terms introduced then reused

Compare perplexity on JUST these tokens between:
- Your model
- Pure Mamba baseline
- Pure attention baseline

**If retrieval works:** Your model should beat Mamba on copy-dependent tokens.

---

### Test 4: State Ablation

After processing a sequence:
1. Zero out the GDN state
2. Continue generation
3. Measure perplexity increase

**If state matters:** Perplexity should spike.
**If state is decorative:** No change (model learned to ignore it).

---

### Test 5: Associative Recall Without Training On It

Train on pure language modeling (no synthetic retrieval data).
Then test zero-shot on:

```
A→1, B→2, C→3, D→4, E→5
[filler]
C→?
```

**This is the real test.** Can the architecture generalize to retrieval from LM training alone, or does it need explicit retrieval supervision?

---

### Test 6: Interference Stress Test

Store similar keys:

```
"The red car is fast. The red truck is slow. The red bike is broken."
"What is the red car?"
```

If model outputs "slow" or "broken" — interference is real and orthogonal keys aren't helping.

---

### Test 7: The Nuclear Option — Compare Against Known Working

Take a small transformer (same param count).
Same training data, same compute.
Same test suite.

**If your architecture can't beat or match transformer on retrieval-heavy tasks, it doesn't work. Full stop.**

---

## Test Harness Skeleton

```python
class HonestTestSuite:
    """Tests that can't be gamed."""
    
    def __init__(self, model, baseline_transformer, baseline_mamba):
        self.model = model
        self.baselines = [baseline_transformer, baseline_mamba]
    
    def test_factual_recall(self, corpus, num_facts=100, distances=[100, 500, 1000, 2000]):
        """Extract real facts from corpus, test recall at various distances."""
        pass
    
    def test_entity_tracking(self, num_entities=[5, 10, 20, 50], seq_lens=[512, 1024, 2048]):
        """Scale both dimensions, find failure modes."""
        pass
    
    def test_copy_dependent_ppl(self, corpus):
        """Identify tokens requiring long-range copy, measure PPL specifically on those."""
        pass
    
    def test_state_ablation(self, corpus):
        """Zero state mid-sequence, measure degradation."""
        pass
    
    def test_zero_shot_ar(self):
        """Associative recall WITHOUT training on synthetic AR data."""
        pass
    
    def comparative_summary(self):
        """Side-by-side against baselines. No excuses."""
        pass
```

---

## What We Actually Proved So Far

With the MARKER/CUE synthetic task:
1. **Orthogonal key bank** prevents key collision (mathematical proof, not empirical)
2. **Sparse writes** (β=0 for non-MARKER) prevents state corruption
3. **Bank key queries** in SWA can retrieve from state

**What this does NOT prove:**
- That the model learns WHAT to store without magic tokens
- That it works at real sequence lengths (1K+)
- That it transfers to language modeling
- That it beats or matches a transformer baseline

---

## Before Writing More Code

Run these tests on current v7. 

If it fails — we know exactly what's broken.
If it passes — we've validated something real.
