# =============================================================================
# TRITON DELTA RULE KERNEL - FIXED FOR TRITON 3.6.0
# =============================================================================
#
# Key Triton constraints:
#   - NO __setitem__ (tensor[i,j] = x is illegal)
#   - Must use vectorized block operations
#   - Use broadcasting for outer products
#   - Load/store entire blocks at once
#
# Target: RTX 4050, PyTorch 2.10.0, Triton 3.6.0, CUDA 12.x
# =============================================================================

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time


@triton.jit
def delta_rule_fwd_kernel(
    # Pointers
    K_ptr, V_ptr, Beta_ptr, G_ptr, State_ptr, Out_ptr,
    # Strides for K [B, T, H, K_DIM]
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    # Strides for V [B, T, H, V_DIM]
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    # Strides for Beta/G [B, T, H]
    stride_bg_b, stride_bg_t, stride_bg_h,
    # Strides for State [B, H, K_DIM, V_DIM]
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    # Strides for Out [B, T, H, V_DIM]
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    # Dimensions
    T,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """
    Delta Rule forward kernel.
    
    Each program instance handles one (batch, head) pair.
    Processes T tokens sequentially, but with vectorized GPU operations.
    
    State update: S = g*S + β*(v - S@k)⊗k
    """
    # Program indices
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Create offset vectors for block operations
    k_offs = tl.arange(0, K_DIM)  # [K_DIM]
    v_offs = tl.arange(0, V_DIM)  # [V_DIM]
    
    # Compute state pointer base and create 2D block pointers
    state_base = State_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    # state_ptrs[i, j] points to State[b, h, i, j]
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    
    # Load initial state as 2D block [K_DIM, V_DIM]
    state = tl.load(state_ptrs).to(tl.float32)
    
    # Process each token
    for t in range(T):
        # === Load k[b, t, h, :] as 1D block [K_DIM] ===
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        # === Load v[b, t, h, :] as 1D block [V_DIM] ===
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        # === Load scalars beta[b, t, h] and g[b, t, h] ===
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        # === Prediction: state @ k_t -> [V_DIM] ===
        # pred[v] = sum_k state[k, v] * k_t[k]
        # Broadcasting: state [K_DIM, V_DIM] * k_t[:, None] [K_DIM, 1] -> [K_DIM, V_DIM]
        # Then sum over axis 0 -> [V_DIM]
        pred = tl.sum(state * k_t[:, None], axis=0)
        
        # === Error ===
        error = v_t - pred  # [V_DIM]
        
        # === Outer product: k_t ⊗ error -> [K_DIM, V_DIM] ===
        # Broadcasting: k_t[:, None] [K_DIM, 1] * error[None, :] [1, V_DIM] -> [K_DIM, V_DIM]
        outer = k_t[:, None] * error[None, :]
        
        # === Update state ===
        state = g_t * state + beta_t * outer
        
        # === Output: retrieve from updated state ===
        out_t = tl.sum(state * k_t[:, None], axis=0)  # [V_DIM]
        
        # === Store output ===
        out_base = Out_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        tl.store(out_base + v_offs * stride_o_v, out_t)
    
    # Store final state
    tl.store(state_ptrs, state)


def triton_delta_rule(k, v, beta, g, initial_state=None):
    """
    Triton-accelerated Delta Rule.
    
    Args:
        k: [B, T, H, K_DIM] - normalized keys
        v: [B, T, H, V_DIM] - values
        beta: [B, T, H] - write gate
        g: [B, T, H] - forget gate
        initial_state: [B, H, K_DIM, V_DIM] or None
    
    Returns:
        output: [B, T, H, V_DIM]
        final_state: [B, H, K_DIM, V_DIM]
    """
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    orig_dtype = k.dtype
    
    # Ensure contiguous and float32 for kernel
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    
    # Initialize or clone state
    if initial_state is None:
        state = torch.zeros(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    else:
        state = initial_state.contiguous().float().clone()
    
    # Allocate output
    out = torch.empty(B, T, H, V_DIM, device=device, dtype=torch.float32)
    
    # Launch kernel: one program per (batch, head)
    grid = (B, H)
    
    delta_rule_fwd_kernel[grid](
        k, v, beta, g, state, out,
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Beta/G strides (same layout)
        beta.stride(0), beta.stride(1), beta.stride(2),
        # State strides
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # Out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Dimensions
        T, K_DIM, V_DIM,
    )
    
    return out.to(orig_dtype), state.to(orig_dtype)


# =============================================================================
# REFERENCE IMPLEMENTATION (for correctness testing)
# =============================================================================

def sequential_delta_rule(k, v, beta, g, initial_state=None):
    """Pure PyTorch sequential - baseline for correctness."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    device, dtype = k.device, k.dtype
    
    if initial_state is None:
        state = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    else:
        state = initial_state.clone()
    
    outputs = []
    for t in range(T):
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t]
        g_t = g[:, t]
        
        pred = torch.einsum('bhkv,bhk->bhv', state, k_t)
        error = v_t - pred
        update = torch.einsum('bhv,bhk->bhkv', error, k_t)
        state = g_t[..., None, None] * state + beta_t[..., None, None] * update
        out_t = torch.einsum('bhkv,bhk->bhv', state, k_t)
        outputs.append(out_t)
    
    return torch.stack(outputs, dim=1), state


# =============================================================================
# TESTING
# =============================================================================

def test_correctness():
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)
    
    B, T, H, K, V = 2, 64, 4, 32, 64
    device = "cuda"
    
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device) - 2)
    g = torch.sigmoid(torch.randn(B, T, H, device=device) + 2)
    
    # Reference
    out_ref, state_ref = sequential_delta_rule(k, v, beta, g)
    
    # Triton
    out_tri, state_tri = triton_delta_rule(k, v, beta, g)
    
    out_err = (out_tri.float() - out_ref.float()).norm() / out_ref.float().norm()
    state_err = (state_tri.float() - state_ref.float()).norm() / state_ref.float().norm()
    
    print(f"Output error:  {out_err.item():.8f}")
    print(f"State error:   {state_err.item():.8f}")
    print(f"→ {'✓ PASS' if out_err < 1e-5 else '✗ FAIL'}")
    
    return out_err.item() < 1e-5


def test_speed():
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK")
    print("=" * 60)
    
    configs = [
        (4, 64, 8, 32, 64),
        (8, 128, 8, 32, 64),
        (8, 256, 8, 32, 64),
        (16, 128, 8, 32, 64),
    ]
    
    n_warmup = 5
    n_runs = 20
    
    print(f"{'Config':>25} | {'PyTorch':>10} | {'Triton':>10} | {'Speedup':>8}")
    print("-" * 60)
    
    for B, T, H, K, V in configs:
        k = F.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1)
        v = torch.randn(B, T, H, V, device='cuda')
        beta = torch.sigmoid(torch.randn(B, T, H, device='cuda') - 2)
        g = torch.sigmoid(torch.randn(B, T, H, device='cuda') + 2)
        
        # Warmup PyTorch
        for _ in range(n_warmup):
            sequential_delta_rule(k, v, beta, g)
            torch.cuda.synchronize()
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            sequential_delta_rule(k, v, beta, g)
        torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - start) / n_runs * 1000
        
        # Warmup Triton
        for _ in range(n_warmup):
            triton_delta_rule(k, v, beta, g)
            torch.cuda.synchronize()
        
        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            triton_delta_rule(k, v, beta, g)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / n_runs * 1000
        
        speedup = pytorch_ms / triton_ms
        config_str = f"B={B},T={T},H={H}"
        print(f"{config_str:>25} | {pytorch_ms:>8.2f}ms | {triton_ms:>8.2f}ms | {speedup:>7.2f}x")


def test_delta_rule_correctness():
    """Verify Delta Rule suppresses redundant writes."""
    print("\n" + "=" * 60)
    print("DELTA RULE VALIDATION")
    print("=" * 60)
    
    B, H, K, V = 1, 4, 32, 64
    device = 'cuda'
    
    state = torch.zeros(B, H, K, V, device=device)
    k = F.normalize(torch.randn(B, H, K, device=device), dim=-1)
    v = torch.randn(B, H, V, device=device)
    
    # First write
    pred1 = torch.einsum('bhkv,bhk->bhv', state, k)
    error1 = v - pred1
    state = state + torch.einsum('bhv,bhk->bhkv', error1, k)
    norm1 = state.norm().item()
    
    # Second write (SAME k, v)
    pred2 = torch.einsum('bhkv,bhk->bhv', state, k)
    error2 = v - pred2
    state = state + torch.einsum('bhv,bhk->bhkv', error2, k)
    norm2 = state.norm().item()
    
    print(f"Error1: {error1.norm().item():.4f}")
    print(f"Error2: {error2.norm().item():.8f} (should be ~0)")
    print(f"State growth: {norm2/norm1:.6f}x (should be ~1.0)")
    print(f"→ {'✓ PASS' if error2.norm().item() < 1e-5 else '✗ FAIL'}")


if __name__ == "__main__":
    test_correctness()
    test_speed()
    test_delta_rule_correctness()
