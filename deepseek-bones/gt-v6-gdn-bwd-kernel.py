
# Forward: S_t = g_t * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) ⊗ k_t
# Output: out_t = S_t @ k_t
#
# Gradients:
#   dL/dv_t = beta_t * (dL/dS_t) @ k_t
#   dL/dbeta_t = (dL/dS_t) : (k_t ⊗ error_t)
#   dL/dg_t = (dL/dS_t) : S_{t-1}
#   dL/dk_t = complex (3 terms): 
#     1. From output: dout_t @ S_t
#     2. From state update: beta_t * (dL/dS_t) @ error_t
#     3. From prediction: -beta_t * k_t * (k_t @ dL/dS_t @ k_t)
#   dL/dS_{t-1} = g_t * dL/dS_t - beta_t * k_t ⊗ (dL/dS_t @ k_t)
#
# Simple forward-backward
# output, final_state = delta_rule_autograd(k, v, beta, g, initial_state)
#
# Compute loss
# loss = output.sum() + final_state.sum()
#
# Backward - automatically computes all gradients
# loss.backward()
#
# Gradients are now in k.grad, v.grad, beta.grad, g.grad, initial_state.grad
# =============================================================================
# DELTA RULE BACKWARD KERNEL - Full Gradient Implementation
# =============================================================================

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def delta_rule_bwd_kernel(
    # Forward inputs (needed for gradients)
    K_ptr, V_ptr, Beta_ptr, G_ptr, State_in_ptr, Out_ptr,
    # Gradients coming from upstream
    dOut_ptr, dState_out_ptr,
    # Output gradients (to compute)
    dK_ptr, dV_ptr, dBeta_ptr, dG_ptr, dState_in_ptr,
    # Strides for forward inputs
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    # Strides for gradients (same layout as forward)
    stride_dk_b, stride_dk_t, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_t, stride_dv_h, stride_dv_d,
    stride_dbg_b, stride_dbg_t, stride_dbg_h,
    stride_ds_b, stride_ds_h, stride_ds_k, stride_ds_v,
    # Dimensions
    T,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    # Options
    NEED_dK: tl.constexpr,
    NEED_dV: tl.constexpr,
    NEED_dBeta: tl.constexpr,
    NEED_dG: tl.constexpr,
):
    """
    Delta Rule backward kernel.
    
    Computes gradients for:
      dK, dV, dBeta, dG, dState_in
    
    Uses the adjoint method: process time steps in reverse.
    Each program handles one (batch, head) pair.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Offset vectors for block operations
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    # =========================================================================
    # 1. FORWARD PASS: Recompute and store necessary intermediate values
    # =========================================================================
    
    # Allocate storage for intermediate values (using shared memory or registers)
    # We need to store: k_t, v_t, beta_t, g_t, state_t, pred_t, error_t for all t
    
    # Shared memory for storing forward pass intermediates
    # Note: For simplicity, we'll recompute on-the-fly in backward pass
    # But for efficiency, we could store in shared memory
    
    # Load initial state
    state_base = State_in_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    state_t = tl.load(state_ptrs).to(tl.float32)
    
    # =========================================================================
    # 2. BACKWARD PASS: Process time steps in reverse
    # =========================================================================
    
    # Initialize gradient accumulators
    dstate = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)
    
    # Initialize output gradients
    if NEED_dK:
        dk_accum = tl.zeros((T, K_DIM), dtype=tl.float32)
    if NEED_dV:
        dv_accum = tl.zeros((T, V_DIM), dtype=tl.float32)
    if NEED_dBeta:
        dbeta_accum = tl.zeros(T, dtype=tl.float32)
    if NEED_dG:
        dg_accum = tl.zeros(T, dtype=tl.float32)
    
    # Process time steps from T-1 down to 0
    for t_rev in range(T):
        t = T - 1 - t_rev  # Reverse order
        
        # =====================================================================
        # Load forward pass values for time t
        # =====================================================================
        
        # Load k[t]
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        # Load v[t]
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        # Load beta[t], g[t]
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        # Load output gradient dOut[t]
        dout_base = dOut_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        dout_t = tl.load(dout_base + v_offs * stride_o_v).to(tl.float32)
        
        # =====================================================================
        # Recompute forward values needed for backward
        # =====================================================================
        
        # Forward prediction at time t: pred_t = state @ k_t
        pred_t = tl.sum(state_t * k_t[:, None], axis=0)  # [V_DIM]
        
        # Forward error at time t: error_t = v_t - pred_t
        error_t = v_t - pred_t  # [V_DIM]
        
        # Forward outer product: outer_t = k_t ⊗ error_t
        outer_t = k_t[:, None] * error_t[None, :]  # [K_DIM, V_DIM]
        
        # =====================================================================
        # Compute gradients
        # =====================================================================
        
        # 1. Gradient from output loss: dL/d(state_t) += dout_t ⊗ k_t
        dstate_from_output = k_t[:, None] * dout_t[None, :]  # [K_DIM, V_DIM]
        dstate += dstate_from_output
        
        # 2. Gradient for v[t]: dL/dv_t = dstate : (beta_t * k_t ⊗ I)
        #    dv_t = beta_t * (dstate @ k_t)
        if NEED_dV:
            dv_t = beta_t * tl.sum(dstate * k_t[:, None], axis=0)  # [V_DIM]
            dv_accum = tl.store(dv_accum, dv_t, t)  # Store at position t
        
        # 3. Gradient for beta[t]: dL/dbeta_t = dstate : outer_t
        if NEED_dBeta:
            dbeta_t = tl.sum(dstate * outer_t)
            dbeta_accum = tl.store(dbeta_accum, dbeta_t, t)
        
        # 4. Gradient for g[t]: dL/dg_t = dstate : state_prev
        if NEED_dG:
            dg_t = tl.sum(dstate * state_t)
            dg_accum = tl.store(dg_accum, dg_t, t)
        
        # 5. Gradient for k[t]: Most complex term
        if NEED_dK:
            # Part 1: From output loss: dL/dk_t from output = dout_t @ state_t
            dk_from_output = tl.sum(dout_t[None, :] * state_t, axis=1)  # [K_DIM]
            
            # Part 2: From state update: dL/dk_t from state = beta_t * (dstate @ error_t)
            dk_from_state = beta_t * tl.sum(dstate * error_t[None, :], axis=1)  # [K_DIM]
            
            # Part 3: From prediction in next step (implicit in dstate): -beta_t * k_t * (dstate : (k_t ⊗ k_t))
            # This is a rank-1 correction: -beta_t * (k_t ⊗ k_t) : dstate * k_t
            # Equivalent to: -beta_t * (k_t @ dstate @ k_t) * k_t
            dstate_k = tl.sum(dstate * k_t[:, None], axis=0)  # [V_DIM]
            k_dstate_k = tl.sum(k_t * tl.sum(dstate_k[None, :] * k_t[:, None], axis=1))  # scalar
            dk_from_pred = -beta_t * k_dstate_k * k_t  # [K_DIM]
            
            dk_t = dk_from_output + dk_from_state + dk_from_pred
            dk_accum = tl.store(dk_accum, dk_t, t)
        
        # 6. Prepare dstate for previous time step
        # dstate for t-1 = g_t * dstate + additional terms from chain rule
        
        # Main term: dstate propagates through g_t
        dstate_prev = g_t * dstate
        
        # Additional term from the prediction: -beta_t * k_t ⊗ (dstate @ k_t)
        dstate_k = tl.sum(dstate * k_t[:, None], axis=0)  # [V_DIM]
        dstate_pred = -beta_t * k_t[:, None] * dstate_k[None, :]  # [K_DIM, V_DIM]
        
        dstate = dstate_prev + dstate_pred
        
        # 7. Update state_t for previous iteration (reverse time)
        # state_{t-1} = (state_t - beta_t * outer_t) / g_t
        # (Numerically stable version)
        if g_t > 1e-8:
            state_t = (state_t - beta_t * outer_t) / g_t
        else:
            state_t = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)
    
    # =========================================================================
    # 3. Store final gradients
    # =========================================================================
    
    # Store dState_in (gradient w.r.t initial state)
    dstate_in_base = dState_in_ptr + pid_b * stride_ds_b + pid_h * stride_ds_h
    dstate_in_ptrs = dstate_in_base + k_offs[:, None] * stride_ds_k + v_offs[None, :] * stride_ds_v
    tl.store(dstate_in_ptrs, dstate)
    
    # Store dK gradients
    if NEED_dK:
        for t in range(T):
            dk_base = dK_ptr + pid_b * stride_dk_b + t * stride_dk_t + pid_h * stride_dk_h
            dk_t = tl.load(dk_accum, t)  # Load from our storage
            tl.store(dk_base + k_offs * stride_dk_d, dk_t)
    
    # Store dV gradients
    if NEED_dV:
        for t in range(T):
            dv_base = dV_ptr + pid_b * stride_dv_b + t * stride_dv_t + pid_h * stride_dv_h
            dv_t = tl.load(dv_accum, t)
            tl.store(dv_base + v_offs * stride_dv_d, dv_t)
    
    # Store dBeta gradients
    if NEED_dBeta:
        for t in range(T):
            dbeta_base = dBeta_ptr + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
            dbeta_t = tl.load(dbeta_accum, t)
            tl.store(dbeta_base, dbeta_t)
    
    # Store dG gradients
    if NEED_dG:
        for t in range(T):
            dg_base = dG_ptr + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
            dg_t = tl.load(dg_accum, t)
            tl.store(dg_base, dg_t)


# =============================================================================
# PYTHON WRAPPER FOR BACKWARD PASS
# =============================================================================

def triton_delta_rule_backward(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    d_out: torch.Tensor,
    d_state_out: torch.Tensor,
    need_gradients: tuple = (True, True, True, True, True)
) -> tuple:
    """
    Backward pass for Delta Rule.
    
    Args:
        k: [B, T, H, K_DIM] - forward keys
        v: [B, T, H, V_DIM] - forward values
        beta: [B, T, H] - forward beta gates
        g: [B, T, H] - forward g gates
        initial_state: [B, H, K_DIM, V_DIM] - initial state from forward
        d_out: [B, T, H, V_DIM] - gradient w.r.t output
        d_state_out: [B, H, K_DIM, V_DIM] - gradient w.r.t final state
        need_gradients: tuple of (need_dk, need_dv, need_dbeta, need_dg, need_dstate_in)
        
    Returns:
        tuple: (d_k, d_v, d_beta, d_g, d_state_in)
    """
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    
    # Ensure contiguous
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    initial_state = initial_state.contiguous().float()
    d_out = d_out.contiguous().float()
    d_state_out = d_state_out.contiguous().float()
    
    # Parse which gradients we need
    need_dk, need_dv, need_dbeta, need_dg, need_dstate_in = need_gradients
    
    # Allocate output gradients
    d_k = torch.empty_like(k) if need_dk else None
    d_v = torch.empty_like(v) if need_dv else None
    d_beta = torch.empty_like(beta) if need_dbeta else None
    d_g = torch.empty_like(g) if need_dg else None
    d_state_in = torch.empty_like(initial_state) if need_dstate_in else None
    
    # Fill with zeros if not needed (kernel won't write to them)
    if not need_dk:
        d_k = torch.zeros_like(k)
    if not need_dv:
        d_v = torch.zeros_like(v)
    if not need_dbeta:
        d_beta = torch.zeros_like(beta)
    if not need_dg:
        d_g = torch.zeros_like(g)
    if not need_dstate_in:
        d_state_in = torch.zeros_like(initial_state)
    
    # Launch kernel
    grid = (B, H)
    
    delta_rule_bwd_kernel[grid](
        # Forward inputs
        k, v, beta, g, initial_state, None,  # Out_ptr not needed in backward
        # Gradients from upstream
        d_out, d_state_out,
        # Output gradients
        d_k, d_v, d_beta, d_g, d_state_in,
        # Strides for forward
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        initial_state.stride(0), initial_state.stride(1), 
        initial_state.stride(2), initial_state.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        # Strides for gradients (assume same as forward)
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_beta.stride(0), d_beta.stride(1), d_beta.stride(2),
        d_state_in.stride(0), d_state_in.stride(1), 
        d_state_in.stride(2), d_state_in.stride(3),
        # Dimensions
        T, K_DIM, V_DIM,
        # Which gradients to compute
        need_dk, need_dv, need_dbeta, need_dg,
    )
    
    # Return only the requested gradients
    result = []
    if need_dk:
        result.append(d_k.to(k.dtype))
    if need_dv:
        result.append(d_v.to(v.dtype))
    if need_dbeta:
        result.append(d_beta.to(beta.dtype))
    if need_dg:
        result.append(d_g.to(g.dtype))
    if need_dstate_in:
        result.append(d_state_in.to(initial_state.dtype))
    
    return tuple(result)


# =============================================================================
# AUTODIFF WRAPPER (PyTorch autograd Function)
# =============================================================================

class DeltaRuleFunction(torch.autograd.Function):
    """PyTorch autograd wrapper for Delta Rule with Triton kernels."""
    
    @staticmethod
    def forward(ctx, k, v, beta, g, initial_state):
        """
        Forward pass.
        
        Args:
            k: [B, T, H, K_DIM] - keys (should be normalized)
            v: [B, T, H, V_DIM] - values
            beta: [B, T, H] - write gates
            g: [B, T, H] - forget gates
            initial_state: [B, H, K_DIM, V_DIM] or None
            
        Returns:
            output: [B, T, H, V_DIM]
            final_state: [B, H, K_DIM, V_DIM]
        """
        # Save for backward
        ctx.save_for_backward(k, v, beta, g, initial_state)
        
        # Run forward kernel
        output, final_state = triton_delta_rule(k, v, beta, g, initial_state)
        
        # Save output for potential use in backward
        ctx.output = output
        ctx.final_state = final_state
        
        return output, final_state
    
    @staticmethod
    def backward(ctx, d_output, d_final_state):
        """
        Backward pass.
        
        Args:
            d_output: [B, T, H, V_DIM] - gradient w.r.t output
            d_final_state: [B, H, K_DIM, V_DIM] - gradient w.r.t final state
            
        Returns:
            d_k, d_v, d_beta, d_g, d_initial_state
        """
        # Retrieve saved tensors
        k, v, beta, g, initial_state = ctx.saved_tensors
        
        # Call backward kernel
        d_k, d_v, d_beta, d_g, d_initial_state = triton_delta_rule_backward(
            k, v, beta, g, initial_state,
            d_output, d_final_state,
            need_gradients=(True, True, True, True, True)
        )
        
        return d_k, d_v, d_beta, d_g, d_initial_state


# =============================================================================
# SIMPLIFIED API
# =============================================================================

def delta_rule_autograd(k, v, beta, g, initial_state=None):
    """
    User-friendly autograd function for Delta Rule.
    
    Args:
        k: [B, T, H, K_DIM] - normalized keys
        v: [B, T, H, V_DIM] - values
        beta: [B, T, H] - write gates
        g: [B, T, H] - forget gates
        initial_state: [B, H, K_DIM, V_DIM] or None
        
    Returns:
        output: [B, T, H, V_DIM]
        final_state: [B, H, K_DIM, V_DIM]
    """
    if initial_state is None:
        B, T, H, K_DIM = k.shape
        V_DIM = v.shape[-1]
        initial_state = torch.zeros(B, H, K_DIM, V_DIM, 
                                   device=k.device, dtype=k.dtype)
    
    return DeltaRuleFunction.apply(k, v, beta, g, initial_state)


# =============================================================================
# TEST: Gradient correctness
# =============================================================================

def test_gradient_correctness():
    """Verify gradients match PyTorch autograd."""
    print("=" * 60)
    print("GRADIENT CORRECTNESS TEST")
    print("=" * 60)
    
    B, T, H, K, V = 2, 32, 4, 16, 32
    
    # Create random inputs with requires_grad
    k = F.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, V, device='cuda').requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device='cuda') - 2).requires_grad_(True)
    g = torch.sigmoid(torch.randn(B, T, H, device='cuda') + 2).requires_grad_(True)
    state = torch.randn(B, H, K, V, device='cuda').requires_grad_(True)
    
    # Reference implementation (simple PyTorch for gradient check)
    def reference_forward(k, v, beta, g, state):
        B, T, H, K_dim = k.shape
        V_dim = v.shape[-1]
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
            outputs.append(torch.einsum('bhkv,bhk->bhv', state, k_t))
        
        return torch.stack(outputs, dim=1), state
    
    # Forward with reference
    output_ref, state_ref = reference_forward(k, v, beta, g, state)
    
    # Create random loss and backward
    loss_ref = output_ref.sum() + state_ref.sum()
    loss_ref.backward()
    
    # Save gradients
    ref_grads = {
        'k': k.grad.clone(),
        'v': v.grad.clone(),
        'beta': beta.grad.clone(),
        'g': g.grad.clone(),
        'state': state.grad.clone()
    }
    
    # Zero gradients
    k.grad = None
    v.grad = None
    beta.grad = None
    g.grad = None
    state.grad = None
    
    # Forward with Triton autograd
    output_tri, state_tri = delta_rule_autograd(k, v, beta, g, state)
    
    # Same loss and backward
    loss_tri = output_tri.sum() + state_tri.sum()
    loss_tri.backward()
    
    # Compare gradients
    print(f"{'Parameter':<10} {'Max Diff':<15} {'Relative Diff':<15}")
    print("-" * 45)
    
    for name, ref_grad in ref_grads.items():
        if name == 'k':
            tri_grad = k.grad
        elif name == 'v':
            tri_grad = v.grad
        elif name == 'beta':
            tri_grad = beta.grad
        elif name == 'g':
            tri_grad = g.grad
        else:
            tri_grad = state.grad
        
        max_diff = (tri_grad - ref_grad).abs().max().item()
        ref_norm = ref_grad.norm().item()
        rel_diff = max_diff / (ref_norm + 1e-8)
        
        print(f"{name:<10} {max_diff:<15.6e} {rel_diff:<15.6e}")
    
    # Check if gradients match
    all_close = all(
        (tri_grad - ref_grad).abs().max().item() < 1e-4
        for tri_grad, ref_grad in zip(
            [k.grad, v.grad, beta.grad, g.grad, state.grad],
            [ref_grads['k'], ref_grads['v'], ref_grads['beta'], ref_grads['g'], ref_grads['state']]
        )
    )
    
    print(f"\n→ {'✓ GRADIENTS MATCH' if all_close else '✗ GRADIENT MISMATCH'}")
    return all_close


if __name__ == "__main__":
    # Run gradient test
    success = test_gradient_correctness()
    
    if success:
        print("\n" + "=" * 60)
        print("BACKWARD KERNEL READY FOR INTEGRATION")
        print("=" * 60)
        print("\nUse: delta_rule_autograd(k, v, beta, g, initial_state)")
        print("Returns: (output, final_state) with full autograd support")
    else:
        print("\nWARNING: Gradient test failed. Kernel needs debugging.")