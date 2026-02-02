"""
Standalone kernel diagnostic - no package import needed.
Just needs the core.py file in same directory or groundthink_v7/ subdirectory.
"""

import torch
import torch.nn.functional as F
import time
import gc
import sys
from pathlib import Path

# Try multiple import paths
try:
    from groundthink_v7.core import chunk_delta_rule, _forward, _backward
    print("✓ Imported from groundthink_v7.core")
except ImportError:
    try:
        from core import chunk_delta_rule, _forward, _backward
        print("✓ Imported from core")
    except ImportError:
        # Add parent to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from groundthink_v7.core import chunk_delta_rule, _forward, _backward
            print("✓ Imported from groundthink_v7.core (via parent path)")
        except ImportError:
            print("✗ Cannot find core.py - place this script alongside core.py or in parent of groundthink_v7/")
            sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def pytorch_sequential(k, v, beta, g, state):
    """Pure PyTorch reference."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    state = state.clone()
    outputs = []
    for t in range(T):
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
        g_t = g[:, t].unsqueeze(-1).unsqueeze(-1)
        pred = torch.einsum('bhkv,bhk->bhv', state, k_t)
        error = v_t - pred
        outer = torch.einsum('bhv,bhk->bhkv', error, k_t)
        state = g_t * state + beta_t * outer
        out_t = torch.einsum('bhkv,bhk->bhv', state, k_t)
        outputs.append(out_t)
    return torch.stack(outputs, dim=1), state


def test_correctness():
    """Verify our kernel matches PyTorch."""
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK")
    print(f"{'='*60}")
    
    device = "cuda"
    B, T, H, K, V = 2, 64, 4, 32, 64
    
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
    state = torch.zeros(B, H, K, V, device=device)
    
    # PyTorch reference
    out_ref, state_ref = pytorch_sequential(k, v, beta, g, state)
    
    # Our kernel
    out_ours, state_ours = chunk_delta_rule(k, v, beta, g, state, chunk_size=32)
    
    out_diff = (out_ours - out_ref).abs().max().item()
    state_diff = (state_ours - state_ref).abs().max().item()
    
    print(f"  Output diff:  {out_diff:.2e}")
    print(f"  State diff:   {state_diff:.2e}")
    print(f"  → {'✓ PASS' if out_diff < 1e-4 and state_diff < 1e-4 else '✗ FAIL'}")
    
    return out_diff < 1e-4


def compare_pytorch(T=128, B=2, H=8, K=32, V=64):
    """Compare speed with PyTorch sequential."""
    print(f"\n{'='*60}")
    print(f"VS PYTORCH SEQUENTIAL: T={T}, B={B}, H={H}")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
    state = torch.zeros(B, H, K, V, device=device)
    
    n_warmup, n_runs = 3, 10
    
    # PyTorch warmup + benchmark
    for _ in range(n_warmup):
        k_ = k.clone().requires_grad_(True)
        v_ = v.clone().requires_grad_(True)
        out, st = pytorch_sequential(k_, v_, beta, g, state)
        (out.sum() + st.sum()).backward()
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        k_ = k.clone().requires_grad_(True)
        v_ = v.clone().requires_grad_(True)
        out, st = pytorch_sequential(k_, v_, beta, g, state)
        (out.sum() + st.sum()).backward()
        torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    # Our kernel warmup + benchmark  
    for _ in range(n_warmup):
        k_ = k.clone().requires_grad_(True)
        v_ = v.clone().requires_grad_(True)
        out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size=64)
        (out.sum() + st.sum()).backward()
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        k_ = k.clone().requires_grad_(True)
        v_ = v.clone().requires_grad_(True)
        out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size=64)
        (out.sum() + st.sum()).backward()
        torch.cuda.synchronize()
    ours_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    print(f"  PyTorch: {pytorch_ms:7.2f}ms")
    print(f"  Ours:    {ours_ms:7.2f}ms")
    print(f"  Speedup: {pytorch_ms/ours_ms:.2f}x")
    
    del k, v, beta, g, state
    gc.collect()
    torch.cuda.empty_cache()
    
    return pytorch_ms, ours_ms


def kernel_breakdown(T=256, B=2, H=8, K=32, V=64, chunk_size=64):
    """Measure forward vs backward separately."""
    print(f"\n{'='*60}")
    print(f"KERNEL BREAKDOWN: T={T}, chunk={chunk_size}")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
    state = torch.zeros(B, H, K, V, device=device)
    
    n_runs = 20
    
    # Forward only
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out, final_state, checkpoints = _forward(k, v, beta, g, state, chunk_size)
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    # Backward only
    d_out = torch.randn_like(out)
    d_state = torch.randn_like(final_state)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dk, dv, dbeta, dg, dstate = _backward(k, v, beta, g, checkpoints, d_out, d_state, chunk_size)
        torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    n_chunks = (T + chunk_size - 1) // chunk_size
    
    print(f"  Forward:  {fwd_ms:6.2f}ms")
    print(f"  Backward: {bwd_ms:6.2f}ms")
    print(f"  Ratio:    {bwd_ms/fwd_ms:.2f}x (expect ~2x)")
    print(f"  Chunks:   {n_chunks}")
    
    if bwd_ms/fwd_ms > 3:
        print(f"  ⚠ Backward unexpectedly slow!")
    
    del k, v, beta, g, state, out, checkpoints
    gc.collect()
    torch.cuda.empty_cache()
    
    return fwd_ms, bwd_ms


def chunk_size_sweep(T=512, B=2, H=8, K=32, V=64):
    """Test different chunk sizes."""
    print(f"\n{'='*60}")
    print(f"CHUNK SIZE SWEEP: T={T}")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device))
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
    state = torch.zeros(B, H, K, V, device=device)
    
    print(f"\n{'Chunk':>6} | {'Fwd+Bwd':>10} | {'Ckpts':>6}")
    print("-" * 35)
    
    for chunk_size in [16, 32, 64, 128, 256]:
        if chunk_size > T:
            continue
            
        gc.collect()
        torch.cuda.empty_cache()
        
        n_warmup, n_runs = 2, 10
        
        # Warmup
        for _ in range(n_warmup):
            k_ = k.clone().requires_grad_(True)
            v_ = v.clone().requires_grad_(True)
            out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size)
            (out.sum() + st.sum()).backward()
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            k_ = k.clone().requires_grad_(True)
            v_ = v.clone().requires_grad_(True)
            out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size)
            (out.sum() + st.sum()).backward()
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) / n_runs * 1000
        
        n_ckpts = (T + chunk_size - 1) // chunk_size + 1
        print(f"{chunk_size:>6} | {total_ms:>9.2f}ms | {n_ckpts:>6}")
    
    del k, v, beta, g, state
    gc.collect()
    torch.cuda.empty_cache()


def sequence_scaling(chunk_size=64, B=2, H=8, K=32, V=64):
    """Test how time scales with T."""
    print(f"\n{'='*60}")
    print(f"SEQUENCE SCALING: chunk={chunk_size}")
    print(f"{'='*60}")
    
    device = "cuda"
    
    print(f"\n{'T':>6} | {'Fwd+Bwd':>10} | {'us/token':>10}")
    print("-" * 40)
    
    for T in [64, 128, 256, 512, 1024, 2048]:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            torch.manual_seed(42)
            k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
            v = torch.randn(B, T, H, V, device=device)
            beta = torch.sigmoid(torch.randn(B, T, H, device=device))
            g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2.0)
            state = torch.zeros(B, H, K, V, device=device)
            
            n_warmup, n_runs = 2, 5
            
            # Warmup
            for _ in range(n_warmup):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                out, st = chunk_delta_rule(k_, v_, beta, g, state, chunk_size)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) / n_runs * 1000
            
            us_per_token = total_ms / T * 1000
            print(f"{T:>6} | {total_ms:>9.2f}ms | {us_per_token:>9.2f}us")
            
            del k, v, beta, g, state
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{T:>6} | OOM")
                break
            raise
    
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_correctness()
    compare_pytorch(T=128)
    kernel_breakdown(T=256)
    chunk_size_sweep(T=512)
    sequence_scaling(chunk_size=64)