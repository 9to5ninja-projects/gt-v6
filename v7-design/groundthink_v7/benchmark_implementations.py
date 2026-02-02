"""
Benchmark: core.py vs parallel_scan_delta_v2.py

Tests:
1. Correctness (forward + backward match PyTorch reference)
2. Memory usage at various sequence lengths
3. Performance (time per forward+backward)
4. Maximum achievable sequence length before OOM

Goal: Determine if parallel_scan_delta_v2 enables longer context.
"""

import torch
import torch.nn.functional as F
import gc
import time
from typing import Tuple, Optional

# Import implementations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core import chunk_delta_rule as current_impl
from core import CHUNK_SIZE

# Try to import parallel scan
try:
    from parallel_scan_delta_v2 import parallel_scan_delta_rule as parallel_impl
    HAS_PARALLEL = True
except ImportError as e:
    print(f"⚠️  Could not import parallel_scan_delta_v2: {e}")
    HAS_PARALLEL = False


def pytorch_reference(k, v, beta, g, initial_state=None):
    """Pure PyTorch reference implementation for correctness testing."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    device = k.device
    
    if initial_state is None:
        state = torch.zeros(B, H, K, V, device=device, dtype=k.dtype)
    else:
        state = initial_state.clone()
    
    outputs = []
    for t in range(T):
        # Delta Rule: state = g * state + beta * k ⊗ (v - state @ k)
        pred = torch.einsum('bhkv,bhk->bhv', state, k[:, t])
        error = v[:, t] - pred
        outer = torch.einsum('bhv,bhk->bhkv', error, k[:, t])
        state = g[:, t].unsqueeze(-1).unsqueeze(-1) * state + \
                beta[:, t].unsqueeze(-1).unsqueeze(-1) * outer
        out_t = torch.einsum('bhkv,bhk->bhv', state, k[:, t])
        outputs.append(out_t)
    
    return torch.stack(outputs, dim=1), state


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_peak_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


def test_correctness(impl_fn, name, B=2, T=128, H=8, K=32, V=64):
    """Test implementation correctness against PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"CORRECTNESS TEST: {name}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create inputs
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
    
    # Test forward
    k1, v1, beta1, g1 = [x.clone().requires_grad_(True) for x in [k, v, beta, g]]
    out1, st1 = impl_fn(k1, v1, beta1, g1, None)
    
    # Reference
    k2, v2, beta2, g2 = [x.clone().requires_grad_(True) for x in [k, v, beta, g]]
    out2, st2 = pytorch_reference(k2, v2, beta2, g2, None)
    
    fwd_out_diff = (out1.float() - out2.float()).abs().max().item()
    fwd_st_diff = (st1.float() - st2.float()).abs().max().item()
    
    print(f"Forward output diff: {fwd_out_diff:.2e}")
    print(f"Forward state diff:  {fwd_st_diff:.2e}")
    
    # Test backward
    loss1 = out1.sum() + st1.sum()
    loss1.backward()
    
    loss2 = out2.sum() + st2.sum()
    loss2.backward()
    
    dk_diff = (k1.grad.float() - k2.grad.float()).abs().max().item()
    dv_diff = (v1.grad.float() - v2.grad.float()).abs().max().item()
    dbeta_diff = (beta1.grad.float() - beta2.grad.float()).abs().max().item()
    dg_diff = (g1.grad.float() - g2.grad.float()).abs().max().item()
    
    print(f"dk diff:    {dk_diff:.2e}")
    print(f"dv diff:    {dv_diff:.2e}")
    print(f"dbeta diff: {dbeta_diff:.2e}")
    print(f"dg diff:    {dg_diff:.2e}")
    
    # Pass/fail
    fwd_ok = fwd_out_diff < 1e-4 and fwd_st_diff < 1e-4
    bwd_ok = dk_diff < 1e-3 and dv_diff < 1e-3 and dbeta_diff < 1e-3 and dg_diff < 1e-3
    
    print(f"\n{'✓ PASS' if fwd_ok and bwd_ok else '✗ FAIL'}")
    return fwd_ok and bwd_ok


def test_memory_and_speed(impl_fn, name, B=2, T=4096, H=8, K=32, V=64, n_runs=5):
    """Test memory usage and speed at a given sequence length."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    reset_gpu_memory()
    
    # Create inputs
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, device=device).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device)).requires_grad_(True)
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0).requires_grad_(True)
    
    mem_before = get_gpu_memory_mb()
    reset_gpu_memory()
    
    # Warmup
    try:
        out, state = impl_fn(k, v, beta, g, None)
        loss = out.sum() + state.sum()
        loss.backward()
        
        del out, state, loss
        torch.cuda.empty_cache()
        gc.collect()
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {"status": "OOM", "seq_len": T}
        raise
    
    reset_gpu_memory()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        k.grad = None
        v.grad = None
        beta.grad = None
        g.grad = None
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        out, state = impl_fn(k, v, beta, g, None)
        loss = out.sum() + state.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    peak_mem = get_gpu_memory_peak_mb()
    avg_time = sum(times) / len(times) * 1000  # ms
    
    return {
        "status": "OK",
        "seq_len": T,
        "peak_mem_mb": peak_mem,
        "avg_time_ms": avg_time,
        "tokens_per_sec": (B * T) / (avg_time / 1000),
    }


def find_max_seq_len(impl_fn, name, B=2, H=8, K=32, V=64, start_T=4096, step=1024):
    """Find maximum sequence length before OOM."""
    print(f"\n{'='*60}")
    print(f"MAX SEQUENCE LENGTH TEST: {name}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_T = start_T
    
    for T in range(start_T, 65536, step):
        reset_gpu_memory()
        
        try:
            torch.manual_seed(42)
            k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).requires_grad_(True)
            v = torch.randn(B, T, H, V, device=device).requires_grad_(True)
            beta = torch.sigmoid(torch.randn(B, T, H, device=device)).requires_grad_(True)
            g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0).requires_grad_(True)
            
            out, state = impl_fn(k, v, beta, g, None)
            loss = out.sum() + state.sum()
            loss.backward()
            
            mem = get_gpu_memory_peak_mb()
            print(f"  T={T:5d}: OK, peak={mem:.0f}MB")
            max_T = T
            
            del k, v, beta, g, out, state, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  T={T:5d}: OOM")
                break
            raise
    
    print(f"\n  → Max sequence length: {max_T}")
    return max_T


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("="*70)
    print("GROUNDTHINK V7: IMPLEMENTATION BENCHMARK")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test correctness
    print("\n" + "="*70)
    print("PHASE 1: CORRECTNESS")
    print("="*70)
    
    current_ok = test_correctness(current_impl, "core.py (current)")
    
    if HAS_PARALLEL:
        parallel_ok = test_correctness(parallel_impl, "parallel_scan_delta_v2.py")
    else:
        parallel_ok = False
        print("\n⚠️  Skipping parallel_scan_delta_v2.py (import failed)")
    
    # Test memory/speed at various lengths
    print("\n" + "="*70)
    print("PHASE 2: MEMORY & SPEED")
    print("="*70)
    
    seq_lens = [512, 1024, 2048, 4096]
    
    print("\n[core.py (current)]")
    print(f"{'T':>6} {'Status':>8} {'Peak MB':>10} {'Time ms':>10} {'tok/s':>12}")
    print("-" * 50)
    
    for T in seq_lens:
        result = test_memory_and_speed(current_impl, "current", T=T)
        if result["status"] == "OK":
            print(f"{T:>6} {'OK':>8} {result['peak_mem_mb']:>10.0f} {result['avg_time_ms']:>10.1f} {result['tokens_per_sec']:>12.0f}")
        else:
            print(f"{T:>6} {'OOM':>8}")
            break
    
    if HAS_PARALLEL:
        print("\n[parallel_scan_delta_v2.py]")
        print(f"{'T':>6} {'Status':>8} {'Peak MB':>10} {'Time ms':>10} {'tok/s':>12}")
        print("-" * 50)
        
        for T in seq_lens:
            result = test_memory_and_speed(parallel_impl, "parallel", T=T)
            if result["status"] == "OK":
                print(f"{T:>6} {'OK':>8} {result['peak_mem_mb']:>10.0f} {result['avg_time_ms']:>10.1f} {result['tokens_per_sec']:>12.0f}")
            else:
                print(f"{T:>6} {'OOM':>8}")
                break
    
    # Find max sequence length
    print("\n" + "="*70)
    print("PHASE 3: MAX SEQUENCE LENGTH (B=2)")
    print("="*70)
    
    max_current = find_max_seq_len(current_impl, "core.py (current)", start_T=4096, step=1024)
    
    if HAS_PARALLEL and parallel_ok:
        max_parallel = find_max_seq_len(parallel_impl, "parallel_scan_delta_v2.py", start_T=4096, step=1024)
    else:
        max_parallel = 0
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Implementation':<30} {'Correct':>10} {'Max T':>10}")
    print("-" * 50)
    print(f"{'core.py (current)':<30} {'✓' if current_ok else '✗':>10} {max_current:>10}")
    if HAS_PARALLEL:
        print(f"{'parallel_scan_delta_v2.py':<30} {'✓' if parallel_ok else '✗':>10} {max_parallel if parallel_ok else 'N/A':>10}")
    
    if HAS_PARALLEL and parallel_ok and max_parallel > max_current:
        print(f"\n🎉 PARALLEL SCAN ENABLES {max_parallel - max_current} ADDITIONAL TOKENS!")
    elif HAS_PARALLEL and parallel_ok:
        print(f"\n→ No significant memory advantage found.")
    
    return {
        "current": {"correct": current_ok, "max_T": max_current},
        "parallel": {"correct": parallel_ok, "max_T": max_parallel} if HAS_PARALLEL else None,
    }


if __name__ == "__main__":
    results = run_full_benchmark()
