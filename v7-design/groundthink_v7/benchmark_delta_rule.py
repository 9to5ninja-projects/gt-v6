"""
Lightweight Delta Rule Benchmark: Ours vs FLA
Run one config at a time to avoid OOM
"""

import torch
import torch.nn.functional as F
import time
import gc
import sys

# =============================================================================
# IMPORTS
# =============================================================================

print("Loading implementations...")

HAS_FLA = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    HAS_FLA = True
    print("✓ FLA loaded")
except ImportError as e:
    print(f"✗ FLA not available: {e}")

HAS_OURS = False
try:
    from groundthink_v7.core import chunk_delta_rule
    HAS_OURS = True
    print("✓ Our implementation loaded")
except ImportError as e:
    print(f"✗ Our implementation not available: {e}")


# =============================================================================
# WRAPPERS
# =============================================================================

def run_ours(k, v, beta, g, state):
    """Our chunk-recurrent implementation."""
    return chunk_delta_rule(k, v, beta, g, state, chunk_size=64)


def run_fla(k, v, beta, g, state):
    """
    FLA's parallel scan implementation.
    Signature: (q, k, v, g, beta, scale, initial_state, output_final_state, ...)
    FLA expects [B, H, T, D] layout
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    
    # Transpose to FLA layout [B, H, T, D]
    q_fla = k.transpose(1, 2).contiguous()
    k_fla = k.transpose(1, 2).contiguous()
    v_fla = v.transpose(1, 2).contiguous()
    beta_fla = beta.transpose(1, 2).contiguous()
    g_fla = g.transpose(1, 2).contiguous()
    
    # FLA call: (q, k, v, g, beta, ...)
    out_fla, final_state = chunk_gated_delta_rule(
        q_fla, k_fla, v_fla,
        g_fla,      # g comes before beta!
        beta_fla,
        scale=1.0,
        initial_state=state,
        output_final_state=True
    )
    
    # Transpose back to [B, T, H, V]
    out = out_fla.transpose(1, 2).contiguous()
    return out, final_state


# =============================================================================
# BENCHMARK ONE CONFIG
# =============================================================================

def benchmark_single(T, B=2, H=8, K=32, V=64, n_warmup=3, n_runs=10):
    """Benchmark a single sequence length."""
    device = "cuda"
    
    print(f"\n{'='*60}")
    print(f"T={T}, B={B}, H={H}, K={K}, V={V}")
    print(f"{'='*60}")
    
    # Create inputs
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float32), dim=-1)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    g = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32) + 2.0)
    state = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    
    results = {}
    
    # === OUR IMPLEMENTATION ===
    if HAS_OURS:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Warmup
            for _ in range(n_warmup):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                beta_ = beta.clone().requires_grad_(True)
                g_ = g.clone().requires_grad_(True)
                out, st = run_ours(k_, v_, beta_, g_, state)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            
            # Timed forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    out, st = run_ours(k, v, beta, g, state)
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / n_runs * 1000
            
            # Timed forward+backward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                beta_ = beta.clone().requires_grad_(True)
                g_ = g.clone().requires_grad_(True)
                out, st = run_ours(k_, v_, beta_, g_, state)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) / n_runs * 1000
            bwd_ms = total_ms - fwd_ms
            
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            results['ours'] = {'fwd': fwd_ms, 'bwd': bwd_ms, 'mem': mem_mb}
            print(f"  OURS:  Fwd={fwd_ms:6.2f}ms  Bwd={bwd_ms:6.2f}ms  Mem={mem_mb:.0f}MB")
            
        except Exception as e:
            print(f"  OURS:  FAILED - {e}")
    
    # === FLA IMPLEMENTATION ===
    if HAS_FLA:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Warmup
            for _ in range(n_warmup):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                beta_ = beta.clone().requires_grad_(True)
                g_ = g.clone().requires_grad_(True)
                out, st = run_fla(k_, v_, beta_, g_, state)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            
            # Timed forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad():
                    out, st = run_fla(k, v, beta, g, state)
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / n_runs * 1000
            
            # Timed forward+backward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                k_ = k.clone().requires_grad_(True)
                v_ = v.clone().requires_grad_(True)
                beta_ = beta.clone().requires_grad_(True)
                g_ = g.clone().requires_grad_(True)
                out, st = run_fla(k_, v_, beta_, g_, state)
                (out.sum() + st.sum()).backward()
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) / n_runs * 1000
            bwd_ms = total_ms - fwd_ms
            
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            results['fla'] = {'fwd': fwd_ms, 'bwd': bwd_ms, 'mem': mem_mb}
            print(f"  FLA:   Fwd={fwd_ms:6.2f}ms  Bwd={bwd_ms:6.2f}ms  Mem={mem_mb:.0f}MB")
            
        except Exception as e:
            print(f"  FLA:   FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    # === COMPARISON ===
    if 'ours' in results and 'fla' in results:
        speedup_fwd = results['ours']['fwd'] / results['fla']['fwd']
        speedup_bwd = results['ours']['bwd'] / results['fla']['bwd']
        print(f"\n  FLA speedup: Fwd={speedup_fwd:.2f}x  Bwd={speedup_bwd:.2f}x")
        results['speedup_fwd'] = speedup_fwd
        results['speedup_bwd'] = speedup_bwd
    
    # Cleanup
    del k, v, beta, g, state
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test sequence lengths one at a time
    # Start small, increase if no crash
    
    all_results = {}
    
    for T in [64, 128, 256, 512, 1024]:
        try:
            r = benchmark_single(T, B=2, H=8, K=32, V=64)
            all_results[T] = r
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM at T={T}, stopping")
                break
            raise
        
        # Give GPU a breather
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'T':>6} | {'Ours Bwd':>10} | {'FLA Bwd':>10} | {'Speedup':>8}")
    print("-" * 50)
    for T, r in all_results.items():
        ours_bwd = f"{r['ours']['bwd']:.2f}ms" if 'ours' in r else "N/A"
        fla_bwd = f"{r['fla']['bwd']:.2f}ms" if 'fla' in r else "N/A"
        speedup = f"{r.get('speedup_bwd', 0):.2f}x" if 'speedup_bwd' in r else "N/A"
        print(f"{T:>6} | {ours_bwd:>10} | {fla_bwd:>10} | {speedup:>8}")