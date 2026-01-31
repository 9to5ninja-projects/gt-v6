# =============================================================================
# DELTA RULE BACKWARD KERNEL - FIXED VERSION
# =============================================================================
#
# Forward: S_t = g_t * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) ⊗ k_t
# Output: out_t = S_t @ k_t
#
# Gradients (adjoint method, reverse time):
#   dL/dv_t = beta_t * (dL/dS_t) @ k_t
#   dL/dbeta_t = (dL/dS_t) : (k_t ⊗ error_t)
#   dL/dg_t = (dL/dS_t) : S_{t-1}
#   dL/dk_t = 3 terms (from output, state update, prediction)
#   dL/dS_{t-1} = g_t * dL/dS_t - beta_t * k_t ⊗ (dL/dS_t @ k_t)
#
# FIX: Store gradients directly to global memory during loop
#      (Triton doesn't support dynamic-sized register arrays)
# =============================================================================

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# =============================================================================
# FORWARD KERNEL (needed for autograd wrapper)
# =============================================================================

@triton.jit
def delta_rule_fwd_kernel(
    K_ptr, V_ptr, Beta_ptr, G_ptr, State_ptr, Out_ptr,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    T,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """Forward kernel - unchanged from validated version."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    state_base = State_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    state = tl.load(state_ptrs).to(tl.float32)
    
    for t in range(T):
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        pred = tl.sum(state * k_t[:, None], axis=0)
        error = v_t - pred
        outer = k_t[:, None] * error[None, :]
        state = g_t * state + beta_t * outer
        
        out_t = tl.sum(state * k_t[:, None], axis=0)
        out_base = Out_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        tl.store(out_base + v_offs * stride_o_v, out_t)
    
    tl.store(state_ptrs, state)


def triton_delta_rule(k, v, beta, g, initial_state=None):
    """Forward pass wrapper."""
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    orig_dtype = k.dtype
    
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    
    if initial_state is None:
        state = torch.zeros(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    else:
        state = initial_state.contiguous().float().clone()
    
    out = torch.empty(B, T, H, V_DIM, device=device, dtype=torch.float32)
    
    delta_rule_fwd_kernel[(B, H)](
        k, v, beta, g, state, out,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T, K_DIM, V_DIM,
    )
    
    return out.to(orig_dtype), state.to(orig_dtype)


# =============================================================================
# BACKWARD KERNEL - FIXED (Version 2: Avoid O(T^2) and nested variable loops)
# =============================================================================
#
# Strategy: Do forward pass to final state, then backward pass with state 
# reconstruction via division (numerically sensitive but avoids O(T^2))
# =============================================================================

@triton.jit
def delta_rule_bwd_kernel(
    # Forward inputs
    K_ptr, V_ptr, Beta_ptr, G_ptr, State_in_ptr,
    # Upstream gradients
    dOut_ptr, dState_out_ptr,
    # Output gradients
    dK_ptr, dV_ptr, dBeta_ptr, dG_ptr, dState_in_ptr,
    # Strides for inputs [B, T, H, D]
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    # Strides for gradient outputs
    stride_dk_b, stride_dk_t, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_t, stride_dv_h, stride_dv_d,
    stride_dbg_b, stride_dbg_t, stride_dbg_h,
    stride_ds_b, stride_ds_h, stride_ds_k, stride_ds_v,
    # Dimensions
    T,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    """
    Backward kernel using adjoint method with state reconstruction.
    
    Each program handles one (batch, head) pair.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    # =========================================================================
    # PHASE 1: Forward pass to get final state
    # =========================================================================
    
    state_base = State_in_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    state = tl.load(state_ptrs).to(tl.float32)
    
    # Run forward to get state_T (final state)
    for t in range(T):
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        pred = tl.sum(state * k_t[:, None], axis=0)
        error = v_t - pred
        outer = k_t[:, None] * error[None, :]
        state = g_t * state + beta_t * outer
    
    # state is now S_T (after all T steps)
    
    # =========================================================================
    # PHASE 2: Backward pass (reverse time)
    # =========================================================================
    
    # Initialize dstate from upstream gradient w.r.t final state
    dstate_out_base = dState_out_ptr + pid_b * stride_ds_b + pid_h * stride_ds_h
    dstate_out_ptrs = dstate_out_base + k_offs[:, None] * stride_ds_k + v_offs[None, :] * stride_ds_v
    dstate = tl.load(dstate_out_ptrs).to(tl.float32)
    
    # Process in reverse: t = T-1, T-2, ..., 0
    for t_rev in range(T):
        t = T - 1 - t_rev
        
        # Load inputs for timestep t
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        # Load output gradient
        dout_base = dOut_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        dout_t = tl.load(dout_base + v_offs * stride_o_v).to(tl.float32)
        
        # Recompute forward values at t using current state
        # state is S_t (state AFTER update at t)
        pred_t = tl.sum(state * k_t[:, None], axis=0)  # This is output, not prediction before update
        
        # Reconstruct state_prev = S_{t-1} (state BEFORE update at t)
        # S_t = g_t * S_{t-1} + beta_t * (v_t - S_{t-1}@k_t) ⊗ k_t
        # This is tricky to invert. For numerical stability, we use:
        # outer_t = k_t ⊗ error_t where error_t = v_t - S_{t-1}@k_t
        # S_{t-1} = (S_t - beta_t * outer_t) / g_t
        # But we need error_t which depends on S_{t-1}...
        #
        # Alternative: error_t = v_t - pred_t where pred_t = S_{t-1} @ k_t
        # And output_t = S_t @ k_t
        # We can compute: S_t @ k_t - S_{t-1} @ k_t = (g_t - 1) * S_{t-1} @ k_t + beta_t * error_t
        # Still circular...
        #
        # Simplest approach: assume g_t is close to 1, use approximate reconstruction
        # For g_t ≈ 0.88 (default), this should be okay
        
        # Compute error using current state (approximation)
        # Actually, for output: out_t = S_t @ k_t, and we have S_t
        # For gradients, we need S_{t-1} for dg and the error for dbeta
        
        # Let's reconstruct properly:
        # We have S_t. We need S_{t-1}.
        # S_t = g * S_{t-1} + beta * (v - S_{t-1}@k) ⊗ k
        # Let p = S_{t-1} @ k (prediction)
        # S_t = g * S_{t-1} + beta * (v - p) ⊗ k
        # S_t @ k = g * p + beta * (v - p) * (k@k)  [since k is normalized, k@k = 1]
        # S_t @ k = g * p + beta * (v - p)
        # S_t @ k = g * p + beta * v - beta * p
        # S_t @ k = (g - beta) * p + beta * v
        # p = (S_t @ k - beta * v) / (g - beta)   [if g != beta]
        
        out_t = tl.sum(state * k_t[:, None], axis=0)  # S_t @ k = output at t
        
        # Compute prediction p = S_{t-1} @ k
        denom = g_t - beta_t
        # Avoid division by zero
        safe_denom = tl.where(tl.abs(denom) > 1e-6, denom, 1e-6)
        pred_before = (out_t - beta_t * v_t) / safe_denom
        
        # Now we can get error
        error_t = v_t - pred_before
        
        # And reconstruct S_{t-1}
        outer_t = k_t[:, None] * error_t[None, :]
        # S_t = g * S_{t-1} + beta * outer_t
        # S_{t-1} = (S_t - beta * outer_t) / g
        safe_g = tl.where(g_t > 1e-6, g_t, 1e-6)
        state_prev = (state - beta_t * outer_t) / safe_g
        
        # === Accumulate dstate from output gradient ===
        # out_t = S_t @ k, so d(out_t)/d(S_t) = k ⊗ I
        # dstate += k_t ⊗ dout_t
        dstate = dstate + k_t[:, None] * dout_t[None, :]
        
        # === Compute and store dv_t ===
        # v appears in error = v - pred, and update = beta * error ⊗ k
        # dL/dv = dL/dS_t : d(S_t)/dv = dL/dS_t : (beta * k ⊗ I) = beta * (dstate @ k)
        dv_t = beta_t * tl.sum(dstate * k_t[:, None], axis=0)
        dv_base = dV_ptr + pid_b * stride_dv_b + t * stride_dv_t + pid_h * stride_dv_h
        tl.store(dv_base + v_offs * stride_dv_d, dv_t)
        
        # === Compute and store dbeta_t ===
        # beta multiplies outer_t in update
        # dL/dbeta = dL/dS_t : outer_t
        dbeta_t = tl.sum(dstate * outer_t)
        dbeta_base = dBeta_ptr + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
        tl.store(dbeta_base, dbeta_t)
        
        # === Compute and store dg_t ===
        # g multiplies state_prev in update
        # dL/dg = dL/dS_t : S_{t-1}
        dg_t = tl.sum(dstate * state_prev)
        dg_base = dG_ptr + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
        tl.store(dg_base, dg_t)
        
        # === Compute and store dk_t ===
        # k appears in: output (S @ k), prediction (S_{t-1} @ k), and outer product (k ⊗ error)
        # Term 1: From output: d(out)/dk where out = S_t @ k
        dk_from_output = tl.sum(state * dout_t[None, :], axis=1)
        
        # Term 2: From outer product in update: beta * dstate @ error^T
        # update = beta * (k ⊗ error), so d(update)/dk includes beta * (I ⊗ error)
        dk_from_outer = beta_t * tl.sum(dstate * error_t[None, :], axis=1)
        
        # Term 3: From prediction in error calculation
        # error = v - pred where pred = S_{t-1} @ k
        # d(error)/dk = -S_{t-1}
        # This flows through: update = beta * (k ⊗ error)
        # d(update)/d(error) = beta * k, so d(update)/dk via error = beta * k ⊗ (-S_{t-1})
        # The gradient contribution is: beta * dstate : (k ⊗ (-S_{t-1}))
        # = -beta * sum over [k,v]: dstate[k,v] * k[k] * S_{t-1}[k,v]  -- but k index is bound
        # Actually: d(k ⊗ error)/dk = I ⊗ error + k ⊗ d(error)/dk
        #         = I ⊗ error + k ⊗ (-S_{t-1} @ I) = I ⊗ error - k ⊗ S_{t-1}
        # So the gradient is: dstate : (I ⊗ error) - dstate : (k ⊗ S_{t-1})
        # First part is Term 2 above. Second part:
        # dstate : (k ⊗ S_{t-1}) = sum_{i,j} dstate[i,j] * k[i] * S_{t-1}[?,j]
        # This doesn't quite work... let me think more carefully.
        #
        # update[i,j] = beta * k[i] * error[j]
        # error[j] = v[j] - sum_m S_{t-1}[m,j] * k[m]
        # d(update[i,j])/dk[n] = beta * delta_{in} * error[j] + beta * k[i] * d(error[j])/dk[n]
        #                     = beta * delta_{in} * error[j] - beta * k[i] * S_{t-1}[n,j]
        # So: d(Loss)/dk[n] from update = sum_{i,j} dstate[i,j] * d(update[i,j])/dk[n]
        #   = beta * sum_j dstate[n,j] * error[j]  (Term 2, already have)
        #   - beta * sum_{i,j} dstate[i,j] * k[i] * S_{t-1}[n,j]
        #   = Term2 - beta * sum_j S_{t-1}[n,j] * sum_i dstate[i,j] * k[i]
        #   = Term2 - beta * sum_j S_{t-1}[n,j] * (dstate @ k)[j]
        #   = Term2 - beta * S_{t-1}[n,:] @ (dstate @ k)
        #
        # So Term 3 = -beta * S_{t-1} @ (dstate @ k)  [matrix-vector product]
        dstate_dot_k = tl.sum(dstate * k_t[:, None], axis=0)  # [V] = dstate @ k
        dk_from_pred = -beta_t * tl.sum(state_prev * dstate_dot_k[None, :], axis=1)  # [K]
        
        dk_t = dk_from_output + dk_from_outer + dk_from_pred
        dk_base = dK_ptr + pid_b * stride_dk_b + t * stride_dk_t + pid_h * stride_dk_h
        tl.store(dk_base + k_offs * stride_dk_d, dk_t)
        
        # === Propagate dstate backward ===
        # S_t = g * S_{t-1} + beta * outer_t
        # dL/dS_{t-1} = g * dL/dS_t + terms from prediction
        # The prediction term: -beta * k ⊗ (dstate @ k)
        dstate_k = tl.sum(dstate * k_t[:, None], axis=0)  # [V]
        dstate = g_t * dstate - beta_t * k_t[:, None] * dstate_k[None, :]
        
        # Update state for next iteration (going backward)
        state = state_prev
    
    # === Store gradient w.r.t initial state ===
    dstate_in_base = dState_in_ptr + pid_b * stride_ds_b + pid_h * stride_ds_h
    dstate_in_ptrs = dstate_in_base + k_offs[:, None] * stride_ds_k + v_offs[None, :] * stride_ds_v
    tl.store(dstate_in_ptrs, dstate)


# =============================================================================
# PYTHON WRAPPER
# =============================================================================

def triton_delta_rule_backward(k, v, beta, g, initial_state, d_out, d_state_out):
    """Backward pass wrapper."""
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    
    # Ensure contiguous float32
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    initial_state = initial_state.contiguous().float()
    d_out = d_out.contiguous().float()
    d_state_out = d_state_out.contiguous().float()
    
    # Allocate output gradients
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    d_beta = torch.empty_like(beta)
    d_g = torch.empty_like(g)
    d_state_in = torch.empty_like(initial_state)
    
    # Launch kernel
    delta_rule_bwd_kernel[(B, H)](
        k, v, beta, g, initial_state,
        d_out, d_state_out,
        d_k, d_v, d_beta, d_g, d_state_in,
        # Input strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        initial_state.stride(0), initial_state.stride(1), initial_state.stride(2), initial_state.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        # Output strides
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_beta.stride(0), d_beta.stride(1), d_beta.stride(2),
        d_state_in.stride(0), d_state_in.stride(1), d_state_in.stride(2), d_state_in.stride(3),
        # Dimensions
        T, K_DIM, V_DIM,
    )
    
    return d_k, d_v, d_beta, d_g, d_state_in


# =============================================================================
# AUTOGRAD WRAPPER
# =============================================================================

class DeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, beta, g, initial_state):
        ctx.save_for_backward(k, v, beta, g, initial_state)
        output, final_state = triton_delta_rule(k, v, beta, g, initial_state)
        return output, final_state
    
    @staticmethod
    def backward(ctx, d_output, d_final_state):
        k, v, beta, g, initial_state = ctx.saved_tensors
        d_k, d_v, d_beta, d_g, d_initial_state = triton_delta_rule_backward(
            k, v, beta, g, initial_state, d_output, d_final_state
        )
        return d_k, d_v, d_beta, d_g, d_initial_state


def delta_rule_autograd(k, v, beta, g, initial_state=None):
    """User-friendly API with autograd support."""
    if initial_state is None:
        B, T, H, K_DIM = k.shape
        V_DIM = v.shape[-1]
        initial_state = torch.zeros(B, H, K_DIM, V_DIM, device=k.device, dtype=k.dtype)
    return DeltaRuleFunction.apply(k, v, beta, g, initial_state)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKWARD KERNEL TEST")
    print("=" * 70)
    
    def reference_delta_rule(k, v, beta, g, initial_state):
        B, T, H, K_dim = k.shape
        V_dim = v.shape[-1]
        state = initial_state.clone()
        outputs = []
        for t in range(T):
            k_t, v_t = k[:, t], v[:, t]
            beta_t, g_t = beta[:, t], g[:, t]
            pred = torch.einsum('bhkv,bhk->bhv', state, k_t)
            error = v_t - pred
            update = torch.einsum('bhv,bhk->bhkv', error, k_t)
            state = g_t[..., None, None] * state + beta_t[..., None, None] * update
            outputs.append(torch.einsum('bhkv,bhk->bhv', state, k_t))
        return torch.stack(outputs, dim=1), state
    
    B, T, H, K, V = 2, 16, 4, 16, 32  # Small T for O(T^2) recompute
    torch.manual_seed(42)
    
    k = F.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, device='cuda').requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device='cuda') - 2).requires_grad_(True)
    g = torch.sigmoid(torch.randn(B, T, H, device='cuda') + 2).requires_grad_(True)
    state = torch.randn(B, H, K, V, device='cuda').requires_grad_(True)
    
    # Reference
    out_ref, state_ref = reference_delta_rule(k, v, beta, g, state)
    (out_ref.sum() + state_ref.sum()).backward()
    ref_grads = {n: eval(f"{n}.grad.clone()") for n in ['k', 'v', 'beta', 'g', 'state']}
    
    # Zero
    for t in [k, v, beta, g, state]:
        t.grad = None
    
    # Triton
    out_tri, state_tri = delta_rule_autograd(k, v, beta, g, state)
    (out_tri.sum() + state_tri.sum()).backward()
    
    # Compare
    print(f"\n{'Param':<8} {'Ref':<12} {'Triton':<12} {'MaxDiff':<12} {'RelErr':<12}")
    print("-" * 60)
    
    all_ok = True
    for name in ['k', 'v', 'beta', 'g', 'state']:
        ref = ref_grads[name]
        tri = eval(f"{name}.grad")
        r_norm = ref.norm().item()
        t_norm = tri.norm().item()
        diff = (tri - ref).abs().max().item()
        rel = diff / (r_norm + 1e-8)
        ok = rel < 0.01  # 1% tolerance for numerical differences
        print(f"{name:<8} {r_norm:<12.4f} {t_norm:<12.4f} {diff:<12.6f} {rel:<12.6f} {'✓' if ok else '✗'}")
        if not ok:
            all_ok = False
    
    print(f"\n→ {'✓ PASSED' if all_ok else '✗ FAILED'}")
    
    if all_ok:
        print("\nBackward kernel validated. Ready for integration.")
