"""
Reproducible benchmark for multi-needle retrieval.
Run: python benchmark_retrieval.py

This tests whether the GDN+SWA architecture can store and retrieve
multiple (key, value) pairs from a sequence.

Task: MARKER VALUE ... MARKER VALUE ... CUE_0 CUE_1 ...
      Model must predict VALUE_i after CUE_i
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from config import HybridConfig
from model import TransparentHybrid
import random

# ============================================================
# CONFIGURATION - Edit these to test different settings
# ============================================================

@dataclass
class BenchmarkConfig:
    # Model
    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    value_dim: int = 64
    layer_pattern: str = 'GS'  # G=GDN, S=SWA
    key_bank_size: int = 16
    
    # Task
    seq_length: int = 128
    n_needles: int = 4
    vocab_size: int = 270  # 0-199 haystack/values, 250=MARKER, 251-266=CUE
    marker_token: int = 250
    cue_token: int = 251  # CUE_i = cue_token + i
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    train_steps: int = 300
    eval_samples: int = 100
    
    # Reproducibility
    seed: int = 42

# ============================================================
# BENCHMARK CODE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_batch(cfg: BenchmarkConfig, batch_size: int, device: str):
    """Generate a batch of needle-in-haystack sequences."""
    B, T, N = batch_size, cfg.seq_length, cfg.n_needles
    
    # Random haystack
    tokens = torch.randint(0, 200, (B, T), device=device)
    targets = torch.full((B, T), -100, device=device)
    
    # Place MARKER-VALUE pairs and CUEs
    for b in range(B):
        for i in range(N):
            value = torch.randint(0, 200, (1,)).item()
            # MARKER at position 10 + i*10, VALUE follows
            marker_pos = 10 + i * 10
            tokens[b, marker_pos] = cfg.marker_token
            tokens[b, marker_pos + 1] = value
            # CUE at position 80 + i*3, target is VALUE
            cue_pos = 80 + i * 3
            tokens[b, cue_pos] = cfg.cue_token + i
            targets[b, cue_pos] = value
    
    return tokens, targets

def evaluate(model, cfg: BenchmarkConfig, device: str):
    """Evaluate retrieval accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(cfg.eval_samples):
            tokens, targets = make_batch(cfg, 1, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = model(tokens)
            
            # Check each CUE position
            for i in range(cfg.n_needles):
                cue_pos = 80 + i * 3
                pred = logits[0, cue_pos].argmax().item()
                expected = targets[0, cue_pos].item()
                if pred == expected:
                    correct += 1
                total += 1
    
    return correct / total

def run_benchmark(bench_cfg: BenchmarkConfig):
    """Run full benchmark and return results."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(bench_cfg.seed)
    
    # Build model
    model_cfg = HybridConfig(
        d_model=bench_cfg.d_model,
        n_heads=bench_cfg.n_heads,
        head_dim=bench_cfg.head_dim,
        value_dim=bench_cfg.value_dim,
        layer_pattern=bench_cfg.layer_pattern,
        key_bank_size=bench_cfg.key_bank_size,
        vocab_size=bench_cfg.vocab_size,
        marker_token=bench_cfg.marker_token,
        cue_token=bench_cfg.cue_token,
        beta_bias=0.0,
        g_bias=2.0,
        shifted_value=True,
    )
    model = TransparentHybrid(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=bench_cfg.learning_rate)
    
    # Train
    model.train()
    for step in range(bench_cfg.train_steps):
        tokens, targets = make_batch(bench_cfg, bench_cfg.batch_size, device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _, _ = model(tokens)
            loss = F.cross_entropy(
                logits.view(-1, bench_cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            acc = evaluate(model, bench_cfg, device)
            print(f"  Step {step:3d}: loss={loss.item():.3f}, acc={acc:.0%}")
    
    # Final eval
    final_acc = evaluate(model, bench_cfg, device)
    return final_acc

def main():
    print("=" * 60)
    print("MULTI-NEEDLE RETRIEVAL BENCHMARK")
    print("=" * 60)
    
    base_cfg = BenchmarkConfig()
    print(f"\nBase config:")
    print(f"  d_model={base_cfg.d_model}, n_heads={base_cfg.n_heads}")
    print(f"  head_dim={base_cfg.head_dim}, value_dim={base_cfg.value_dim}")
    print(f"  layer_pattern='{base_cfg.layer_pattern}'")
    print(f"  seq_length={base_cfg.seq_length}")
    print(f"  train_steps={base_cfg.train_steps}, batch_size={base_cfg.batch_size}")
    print(f"  seed={base_cfg.seed}")
    
    results = {}
    
    # Test different needle counts
    for n_needles in [1, 2, 4, 8]:
        print(f"\n--- {n_needles} needle(s) ---")
        cfg = BenchmarkConfig(n_needles=n_needles)
        acc = run_benchmark(cfg)
        results[n_needles] = acc
        print(f"  FINAL: {acc:.0%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for n, acc in results.items():
        status = "✓" if acc >= 0.95 else "✗"
        print(f"  {n} needle(s): {acc:.0%} {status}")
    
    return results

if __name__ == "__main__":
    main()
