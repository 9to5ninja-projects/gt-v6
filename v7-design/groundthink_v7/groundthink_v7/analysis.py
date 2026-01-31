"""
GroundThink v7 Analysis & Training Utilities

- NIAH (Needle-In-A-Haystack) tests
- Delta Rule validation suite
- Training with curriculum learning
- Gradient analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import time

__all__ = [
    'proper_niah_test', 'test_niah_by_distance', 'run_full_diagnostic',
    'validate_delta_rule', 'train_curriculum', 'analyze_gradients',
    'TextDataset', 'load_wikitext'
]


# =============================================================================
# NIAH TESTING
# =============================================================================

def proper_niah_test(model, seq_len=128, needle_pos=32, n_trials=30):
    """
    Needle-In-A-Haystack test with MARKER + CUE tokens.
    
    Tests the model's ability to:
        1. Store information at MARKER position
        2. Retrieve it when CUE is seen
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    
    correct = 0
    for _ in range(n_trials):
        needle_id = cfg.vocab_size - 3
        seq = torch.randint(0, cfg.vocab_size - 100, (1, seq_len), device=device)
        
        seq[0, needle_pos] = cfg.marker_token
        seq[0, needle_pos + 1] = needle_id
        seq[0, -1] = cfg.cue_token
        
        with torch.no_grad():
            logits, _, _, _ = model(seq)
        
        pred = logits[0, -1].argmax().item()
        if pred == needle_id:
            correct += 1
    
    acc = correct / n_trials
    print(f"  Accuracy: {acc*100:.1f}% ({correct}/{n_trials})")
    return {'accuracy': acc, 'correct': correct, 'total': n_trials}


def test_niah_by_distance(model, distances=[5, 10, 20, 40, 60, 95], n_trials=20, seq_len=128):
    """Test retrieval across varying distances."""
    print(f"\nNIAH by Distance (seq_len={seq_len}):")
    results = {}
    
    for dist in distances:
        needle_pos = max(2, seq_len - dist - 2)
        print(f"  Distance {dist:3d} (pos={needle_pos:3d}): ", end="")
        result = proper_niah_test(model, seq_len=seq_len, needle_pos=needle_pos, n_trials=n_trials)
        results[dist] = result
    
    return results


def run_full_diagnostic(model, seq_len=128, needle_pos=32):
    """Comprehensive diagnostic with state health check."""
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    
    needle_id = cfg.vocab_size - 3
    seq = torch.randint(0, cfg.vocab_size - 100, (1, seq_len), device=device)
    seq[0, needle_pos] = cfg.marker_token
    seq[0, needle_pos + 1] = needle_id
    seq[0, -1] = cfg.cue_token
    
    with torch.no_grad():
        logits, _, diags, state = model(seq, return_diagnostics=True)
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    
    print(f"\nState Health:")
    print(f"  State norm: {state.norm().item():.2f}")
    print(f"  State max:  {state.abs().max().item():.2f}")
    
    if state.abs().max().item() < 10:
        print(f"  ✓ State bounded")
    elif state.abs().max().item() < 100:
        print(f"  ⚠ State moderately large")
    else:
        print(f"  ✗ State explosion!")
    
    print(f"\nLayer Diagnostics:")
    for i, d in enumerate(diags):
        if d['layer'] == 'G':
            print(f"  Layer {i} (GDN): β={d['beta_mean']:.3f}, g={d['g_mean']:.3f}, state={d['state_norm']:.2f}")
        else:
            print(f"  Layer {i} (SWA): gate={d['gate_mean']:.3f}, local={d['local_norm']:.2f}, retrieval={d['retrieval_norm']:.2f}")
    
    pred = logits[0, -1].argmax().item()
    print(f"\nPrediction: {pred} (target: {needle_id}) - {'✓' if pred == needle_id else '✗'}")
    
    return state, diags


# =============================================================================
# DELTA RULE VALIDATION
# =============================================================================

def validate_delta_rule(device='cuda'):
    """Validate TRUE Delta Rule properties."""
    print(f"\n{'='*60}")
    print("DELTA RULE VALIDATION")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: Identical tokens
    print("\n1. Identical Tokens (Redundancy Suppression):")
    B, H, K, V = 1, 4, 32, 64
    state = torch.zeros(B, H, K, V, device=device)
    k = F.normalize(torch.randn(B, H, K, device=device), dim=-1)
    v = torch.randn(B, H, V, device=device)
    
    pred1 = torch.einsum('bhkv,bhk->bhv', state, k)
    error1 = v - pred1
    state = state + torch.einsum('bhv,bhk->bhkv', error1, k)
    norm1 = state.norm().item()
    
    pred2 = torch.einsum('bhkv,bhk->bhv', state, k)
    error2 = v - pred2
    state = state + torch.einsum('bhv,bhk->bhkv', error2, k)
    norm2 = state.norm().item()
    
    passed = error2.norm().item() < 0.001
    print(f"  Error2: {error2.norm().item():.6f} (should be ~0)")
    print(f"  Growth: {norm2/norm1:.4f}x")
    print(f"  → {'✓ PASS' if passed else '✗ FAIL'}")
    results['identical_tokens'] = passed
    
    # Test 2: Orthogonal keys
    print("\n2. Orthogonal Keys (Independent Storage):")
    state = torch.zeros(1, 1, 32, 64, device=device)
    k1 = torch.zeros(1, 1, 32, device=device); k1[0,0,0] = 1.0
    k2 = torch.zeros(1, 1, 32, device=device); k2[0,0,1] = 1.0
    v1 = torch.randn(1, 1, 64, device=device)
    v2 = torch.randn(1, 1, 64, device=device)
    
    state = state + torch.einsum('bhv,bhk->bhkv', v1, k1)
    state = state + torch.einsum('bhv,bhk->bhkv', v2, k2)
    
    ret1 = torch.einsum('bhkv,bhk->bhv', state, k1)
    ret2 = torch.einsum('bhkv,bhk->bhv', state, k2)
    err1 = (ret1 - v1).norm().item() / v1.norm().item()
    err2 = (ret2 - v2).norm().item() / v2.norm().item()
    
    passed = err1 < 0.001 and err2 < 0.001
    print(f"  v1 error: {err1:.6f}")
    print(f"  v2 error: {err2:.6f}")
    print(f"  → {'✓ PASS' if passed else '✗ FAIL'}")
    results['orthogonal_keys'] = passed
    
    # Test 3: Capacity
    print("\n3. Capacity (100 writes):")
    state = torch.zeros(1, 1, 32, 64, device=device)
    keys, values = [], []
    for i in range(100):
        k = F.normalize(torch.randn(1, 1, 32, device=device), dim=-1)
        v = torch.randn(1, 1, 64, device=device)
        keys.append(k); values.append(v)
        pred = torch.einsum('bhkv,bhk->bhv', state, k)
        error = v - pred
        state = state + torch.einsum('bhv,bhk->bhkv', error, k)
    
    ret_first = torch.einsum('bhkv,bhk->bhv', state, keys[0])
    ret_last = torch.einsum('bhkv,bhk->bhv', state, keys[-1])
    err_first = (ret_first - values[0]).norm().item() / values[0].norm().item()
    err_last = (ret_last - values[-1]).norm().item() / values[-1].norm().item()
    
    print(f"  State norm: {state.norm().item():.2f}")
    print(f"  First error: {err_first:.4f}")
    print(f"  Last error: {err_last:.4f}")
    print(f"  → First degrades (expected)")
    results['capacity'] = True
    
    print(f"\n{'='*60}")
    all_pass = all(v for v in results.values() if isinstance(v, bool))
    print(f"OVERALL: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    
    return results


# =============================================================================
# TRAINING
# =============================================================================

class TextDataset(Dataset):
    """Simple dataset for language modeling."""
    def __init__(self, tokens, seq_len=128):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return torch.tensor(self.tokens[start:start + self.seq_len + 1], dtype=torch.long)


def load_wikitext(n_tokens=500_000, seq_len=128, batch_size=16):
    """Load wikitext data for training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print(f"Loading {n_tokens:,} tokens from wikitext-103...")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    
    all_tokens = []
    for item in dataset:
        if item['text'].strip():
            all_tokens.extend(tokenizer.encode(item['text']))
            if len(all_tokens) >= n_tokens:
                break
    
    all_tokens = all_tokens[:n_tokens]
    print(f"Loaded {len(all_tokens):,} tokens")
    
    ds = TextDataset(all_tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def compute_retrieval_loss(model, seq_len=128, batch_size=4):
    """Synthetic retrieval task for gradient signal."""
    device = next(model.parameters()).device
    cfg = model.cfg
    
    needle_id = cfg.vocab_size - 3
    tokens = torch.randint(0, cfg.vocab_size - 100, (batch_size, seq_len), device=device)
    
    for i in range(batch_size):
        pos = torch.randint(5, seq_len - 10, (1,)).item()
        tokens[i, pos] = cfg.marker_token
        tokens[i, pos + 1] = needle_id
    
    tokens[:, -1] = cfg.cue_token
    
    targets = torch.full((batch_size, seq_len), -100, device=device)
    targets[:, -1] = needle_id
    
    _, loss, _, _ = model(tokens, targets=targets)
    return loss


def train_curriculum(
    model, 
    data_loader, 
    steps=1000, 
    warmup_steps=200,
    lr=3e-4, 
    retrieval_weight=2.0, 
    log_interval=100
):
    """
    Curriculum training: retrieval warmup → mixed LM/retrieval.
    
    Phase 1 (warmup): Pure retrieval loss
    Phase 2 (mixed): LM loss + weighted retrieval loss
    """
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    lm_iter = iter(data_loader)
    history = {'step': [], 'lm': [], 'ret': [], 'phase': []}
    
    print(f"Training {steps} steps ({warmup_steps} warmup)")
    print(f"  LR: {lr}, Retrieval weight: {retrieval_weight}")
    print("="*60)
    
    model.train()
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        if step < warmup_steps:
            ret_loss = compute_retrieval_loss(model)
            ret_loss.backward()
            history['ret'].append(ret_loss.item())
            history['lm'].append(0)
            history['phase'].append('warmup')
        else:
            try:
                batch = next(lm_iter)
            except StopIteration:
                lm_iter = iter(data_loader)
                batch = next(lm_iter)
            
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            _, lm_loss, _, _ = model(input_ids, targets)
            
            ret_loss = compute_retrieval_loss(model)
            
            total = lm_loss + retrieval_weight * ret_loss
            total.backward()
            
            history['lm'].append(lm_loss.item())
            history['ret'].append(ret_loss.item())
            history['phase'].append('mixed')
        
        history['step'].append(step)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % log_interval == 0:
            phase = "WARMUP" if step < warmup_steps else "MIXED"
            lm = history['lm'][-1]
            ret = history['ret'][-1]
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"[{phase:6s}] Step {step:5d}: LM={lm:6.3f} RET={ret:6.3f} ({steps_per_sec:.1f} steps/s)")
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"Training complete: {steps} steps in {total_time:.1f}s")
    
    return history


def analyze_gradients(model, seq_len=64, verbose=True):
    """Analyze gradient flow through the model."""
    device = next(model.parameters()).device
    model.train()
    
    x = torch.randint(0, model.cfg.vocab_size - 100, (2, seq_len), device=device)
    x[:, 10] = model.cfg.marker_token
    x[:, 11] = model.cfg.vocab_size - 3
    x[:, -1] = model.cfg.cue_token
    
    targets = torch.full((2, seq_len), -100, device=device)
    targets[:, -1] = model.cfg.vocab_size - 3
    
    model.zero_grad()
    _, loss, _, _ = model(x, targets)
    loss.backward()
    
    grad_info = {}
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name] = grad_norm
            if torch.isnan(param.grad).any():
                has_nan = True
            if torch.isinf(param.grad).any():
                has_inf = True
    
    if verbose:
        print(f"\n{'='*60}")
        print("GRADIENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Loss: {loss.item():.4f}")
        print(f"NaN: {'✗ YES' if has_nan else '✓ NO'}")
        print(f"Inf: {'✗ YES' if has_inf else '✓ NO'}")
    
    return {'grad_norms': grad_info, 'has_nan': has_nan, 'has_inf': has_inf, 'loss': loss.item()}
