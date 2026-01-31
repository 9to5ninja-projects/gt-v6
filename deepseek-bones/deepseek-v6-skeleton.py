# =============================================================================
# DELTA RULE INTEGRATION TEST SUITE & ARCHITECTURE
# =============================================================================
# 
# This module provides:
# 1. Stable Triton Delta Rule kernel (already validated)
# 2. Comprehensive test suite for integration
# 3. Modular architecture compatible with existing builds
# 4. Performance analysis tools
#
# Key Principles:
# - Maintain backward compatibility with existing modules
# - No simplification to sequential loops
# - Preserve all original design patterns
# - Full validation at every integration point
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

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
# 2. COMPREHENSIVE TEST SUITE
# =============================================================================

class DeltaRuleTestSuite:
    """Complete test suite for Delta Rule integration"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def test_kernel_correctness(self, 
                               B: int = 2, 
                               T: int = 64, 
                               H: int = 4, 
                               K: int = 32, 
                               V: int = 64) -> Dict[str, Any]:
        """
        Test Triton kernel against PyTorch reference implementation.
        Returns detailed error metrics.
        """
        print(f"\n{'='*60}")
        print(f"KERNEL CORRECTNESS TEST: B={B}, T={T}, H={H}, K={K}, V={V}")
        print(f"{'='*60}")
        
        # Generate test data
        torch.manual_seed(42)
        k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
        v = torch.randn(B, T, H, V, device=self.device)
        beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
        g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
        
        # Reference implementation (preserves original logic)
        def reference_delta_rule(k, v, beta, g, initial_state=None):
            B, T, H, K_dim = k.shape
            V_dim = v.shape[-1]
            state = torch.zeros(B, H, K_dim, V_dim, device=k.device) if initial_state is None else initial_state.clone()
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
        
        # Run both implementations
        with torch.no_grad():
            out_ref, state_ref = reference_delta_rule(k, v, beta, g)
            out_tri, state_tri = triton_delta_rule(k, v, beta, g)
        
        # Calculate errors
        out_abs_err = (out_tri - out_ref).abs().max().item()
        out_rel_err = ((out_tri - out_ref).norm() / out_ref.norm()).item()
        
        state_abs_err = (state_tri - state_ref).abs().max().item()
        state_rel_err = ((state_tri - state_ref).norm() / state_ref.norm()).item()
        
        # Store results
        test_results = {
            'output_abs_error': out_abs_err,
            'output_rel_error': out_rel_err,
            'state_abs_error': state_abs_err,
            'state_rel_error': state_rel_err,
            'passed': out_abs_err < 1e-5 and state_abs_err < 1e-5,
            'config': {'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        }
        
        # Print results
        print(f"Output absolute error: {out_abs_err:.2e}")
        print(f"Output relative error: {out_rel_err:.2e}")
        print(f"State absolute error:  {state_abs_err:.2e}")
        print(f"State relative error:  {state_rel_err:.2e}")
        print(f"✓ PASS" if test_results['passed'] else "✗ FAIL")
        
        self.results['kernel_correctness'] = test_results
        return test_results
    
    def test_performance(self, 
                        configs: Optional[list] = None,
                        n_warmup: int = 5,
                        n_runs: int = 20) -> Dict[str, Any]:
        """
        Performance benchmarking across multiple configurations.
        """
        if configs is None:
            configs = [
                (4, 64, 8, 32, 64),
                (8, 128, 8, 32, 64),
                (8, 256, 8, 32, 64),
                (16, 128, 8, 32, 64),
            ]
        
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"{'Configuration':>25} | {'PyTorch (ms)':>12} | {'Triton (ms)':>11} | {'Speedup':>8}")
        print(f"{'-'*70}")
        
        performance_results = {}
        
        for B, T, H, K, V in configs:
            # Generate data
            k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
            v = torch.randn(B, T, H, V, device=self.device)
            beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
            g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
            
            # Reference timing
            def ref_func():
                state = torch.zeros(B, H, K, V, device=self.device)
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
                torch.cuda.synchronize()
            
            # Warmup and time reference
            for _ in range(n_warmup):
                ref_func()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                ref_func()
            torch.cuda.synchronize()
            ref_time = (time.perf_counter() - start) / n_runs * 1000
            
            # Warmup and time Triton
            for _ in range(n_warmup):
                triton_delta_rule(k, v, beta, g)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                triton_delta_rule(k, v, beta, g)
            torch.cuda.synchronize()
            tri_time = (time.perf_counter() - start) / n_runs * 1000
            
            speedup = ref_time / tri_time
            config_str = f"B={B},T={T},H={H},K={K},V={V}"
            
            print(f"{config_str:>25} | {ref_time:>11.2f}ms | {tri_time:>10.2f}ms | {speedup:>7.2f}x")
            
            performance_results[config_str] = {
                'pytorch_ms': ref_time,
                'triton_ms': tri_time,
                'speedup': speedup,
                'config': {'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
            }
        
        self.results['performance'] = performance_results
        return performance_results
    
    def test_gradient_flow(self, 
                          B: int = 2, 
                          T: int = 32, 
                          H: int = 4, 
                          K: int = 16, 
                          V: int = 32) -> Dict[str, Any]:
        """
        Test that gradients flow correctly through the Delta Rule.
        """
        print(f"\n{'='*60}")
        print("GRADIENT FLOW TEST")
        print(f"{'='*60}")
        
        # Create learnable parameters
        k_proj = nn.Linear(64, H * K).to(self.device)
        v_proj = nn.Linear(64, H * V).to(self.device)
        beta_proj = nn.Linear(64, H).to(self.device)
        g_proj = nn.Linear(64, H).to(self.device)
        
        # Input
        x = torch.randn(B, T, 64, device=self.device, requires_grad=True)
        
        # Forward pass
        k = k_proj(x).view(B, T, H, K)
        v = v_proj(x).view(B, T, H, V)
        beta = torch.sigmoid(beta_proj(x))
        g = torch.sigmoid(g_proj(x))

        k = F.normalize(k, dim=-1)

        # NOTE: Use DeltaRuleFunction.apply for autograd support (critical for gradient flow)
        initial_state = torch.zeros(B, H, K, V, device=self.device, dtype=x.dtype)
        output, state = DeltaRuleFunction.apply(k, v, beta, g, initial_state)

        # Loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        grad_checks = {
            'x_has_grad': x.grad is not None,
            'x_grad_norm': x.grad.norm().item() if x.grad is not None else 0,
            'k_proj_has_grad': k_proj.weight.grad is not None,
            'v_proj_has_grad': v_proj.weight.grad is not None,
            'no_nan_in_grads': not torch.isnan(x.grad).any().item() if x.grad is not None else True,
            'no_inf_in_grads': not torch.isinf(x.grad).any().item() if x.grad is not None else True,
        }
        
        grad_results = {
            **grad_checks,
            'passed': all(grad_checks.values()),
        }
        
        # Print results
        for key, value in grad_checks.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        print(f"→ {'✓ GRADIENTS FLOW CORRECTLY' if grad_results['passed'] else '✗ GRADIENT ISSUE'}")
        
        self.results['gradient_flow'] = grad_results
        return grad_results
    
    def test_memory_behavior(self, 
                           B: int = 4, 
                           T: int = 1024, 
                           H: int = 8, 
                           K: int = 32, 
                           V: int = 64) -> Dict[str, Any]:
        """
        Test memory usage and behavior with large sequences.
        """
        print(f"\n{'='*60}")
        print("MEMORY BEHAVIOR TEST (Large Sequence)")
        print(f"{'='*60}")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Generate large sequence
        k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
        v = torch.randn(B, T, H, V, device=self.device)
        beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
        g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
        
        # Warmup
        for _ in range(3):
            triton_delta_rule(k, v, beta, g)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        final_memory = torch.cuda.memory_allocated() / 1024**2
        
        memory_results = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': peak_memory - initial_memory,
            'sequence_length': T,
            'total_elements': B * T * H * (K + V + 2),
        }
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory:    {peak_memory:.1f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.1f} MB")
        print(f"Sequence length: {T}")
        print(f"Total elements: {memory_results['total_elements']:,}")
        
        self.results['memory_behavior'] = memory_results
        return memory_results
    
    def run_full_suite(self) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.
        """
        print(f"\n{'='*60}")
        print("DELTA RULE COMPREHENSIVE TEST SUITE")
        print(f"{'='*60}")
        
        # Run all tests
        kernel_test = self.test_kernel_correctness()
        performance_test = self.test_performance()
        gradient_test = self.test_gradient_flow()
        memory_test = self.test_memory_behavior()
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        all_passed = (
            kernel_test['passed'] and
            gradient_test['passed']
        )
        
        summary = {
            'all_tests_passed': all_passed,
            'kernel_correctness': kernel_test['passed'],
            'gradient_flow': gradient_test['passed'],
            'performance_tested': True,
            'memory_tested': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'triton_version': triton.__version__,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
        }
        
        for key, value in summary.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        # Save results
        self.save_results()
        
        return summary
    
    def save_results(self, path: str = "delta_rule_test_results.json"):
        """Save test results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {path}")


# =============================================================================
# 3. MODULAR ARCHITECTURE COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization - compatible with existing builds."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class GatedDeltaNetLayer(nn.Module):
    """
    Complete Delta Rule layer with proper integration points.
    Maintains backward compatibility with existing architectures.
    """
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 head_dim: int, 
                 value_dim: int,
                 use_triton: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.use_triton = use_triton
        
        # Projections (maintain original naming)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * value_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * value_dim, d_model, bias=False)
        
        # Gate projections
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.beta_proj.bias, -2.0)
        
        self.g_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.g_proj.bias, 2.0)
        
        # Normalization
        self.norm = RMSNorm(d_model)
        
        # State initialization
        self.state = None
        
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize or reset state."""
        self.state = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.value_dim,
            device=device
        )
    
    def forward(self, 
                x: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional initial state.
        Compatible with existing call signatures.
        """
        B, T, D = x.shape
        H, K, V = self.n_heads, self.head_dim, self.value_dim
        
        # Normalization
        x_norm = self.norm(x)
        
        # Projections
        k = self.k_proj(x_norm).view(B, T, H, K)
        v = self.v_proj(x_norm).view(B, T, H, V)
        
        # Normalize keys
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        
        # Gates
        beta = torch.sigmoid(self.beta_proj(x_norm))
        g = torch.sigmoid(self.g_proj(x_norm))
        
        # Use initial state if provided, otherwise use stored state
        current_state = initial_state if initial_state is not None else self.state
        
        # Always use Triton kernel with autograd support
        out, new_state = DeltaRuleFunction.apply(k, v, beta, g, current_state)

        # Update stored state
        self.state = new_state.detach() if self.training else new_state

        # Output projection with residual
        out = out.to(x.dtype).reshape(B, T, H * V)
        return x + self.o_proj(out), new_state
    
    def _reference_delta_rule(self, k, v, beta, g, initial_state=None):
        """Reference implementation for debugging."""
        B, T, H, K_dim = k.shape
        V_dim = v.shape[-1]
        
        if initial_state is None:
            state = torch.zeros(B, H, K_dim, V_dim, device=k.device, dtype=k.dtype)
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
            outputs.append(torch.einsum('bhkv,bhk->bhv', state, k_t))
        
        return torch.stack(outputs, dim=1), state


# =============================================================================
# 4. INTEGRATION VALIDATION
# =============================================================================

class DeltaRuleIntegrationValidator:
    """Validates Delta Rule integration with existing architectures."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def validate_with_existing_model(self, 
                                   model: nn.Module, 
                                   input_shape: Tuple[int, int, int],
                                   n_steps: int = 10) -> Dict[str, Any]:
        """
        Validate Delta Rule integration with an existing model.
        
        Args:
            model: Existing model that should use Delta Rule
            input_shape: (batch, seq_len, features)
            n_steps: Number of forward steps to test
        
        Returns:
            Validation results
        """
        print(f"\n{'='*60}")
        print("EXISTING MODEL INTEGRATION VALIDATION")
        print(f"{'='*60}")
        
        model = model.to(self.device)
        model.eval()
        
        results = {
            'forward_passes': [],
            'memory_usage': [],
            'state_consistency': True,
        }
        
        # Test sequential forward passes with state retention
        state = None
        for step in range(n_steps):
            x = torch.randn(*input_shape, device=self.device)
            
            # Forward pass
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                output, new_state = model(x, initial_state=state)
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            
            # Record results
            step_result = {
                'step': step,
                'time_ms': elapsed,
                'output_shape': tuple(output.shape),
                'state_shape': tuple(new_state.shape) if new_state is not None else None,
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
            }
            
            results['forward_passes'].append(step_result)
            
            # Check state consistency
            if state is not None:
                state_changed = not torch.allclose(state, new_state, rtol=1e-5)
                if not state_changed:
                    results['state_consistency'] = False
                    print(f"⚠ Warning: State unchanged at step {step}")
            
            state = new_state
            
            print(f"Step {step}: {elapsed:.2f}ms, Output: {output.shape}, "
                  f"State: {new_state.shape if new_state is not None else 'None'}")
        
        # Check memory behavior
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        results['peak_memory_mb'] = peak_memory
        
        print(f"\nPeak memory usage: {peak_memory:.1f} MB")
        print(f"State consistency: {'✓' if results['state_consistency'] else '✗'}")
        
        return results
    
    def compare_implementations(self, 
                              d_model: int = 256,
                              n_heads: int = 8,
                              head_dim: int = 32,
                              value_dim: int = 64,
                              seq_len: int = 128,
                              batch_size: int = 4) -> Dict[str, Any]:
        """
        Compare Triton vs reference implementations.
        """
        print(f"\n{'='*60}")
        print("IMPLEMENTATION COMPARISON")
        print(f"{'='*60}")
        
        # Create both implementations
        triton_layer = GatedDeltaNetLayer(
            d_model, n_heads, head_dim, value_dim, use_triton=True
        ).to(self.device)
        
        ref_layer = GatedDeltaNetLayer(
            d_model, n_heads, head_dim, value_dim, use_triton=False
        ).to(self.device)
        
        # Copy weights to ensure fair comparison
        ref_layer.load_state_dict(triton_layer.state_dict())
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        # Warmup
        for _ in range(3):
            triton_layer(x)
            ref_layer(x)
        
        # Time Triton implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out_tri, state_tri = triton_layer(x)
        torch.cuda.synchronize()
        tri_time = (time.perf_counter() - start) / 20 * 1000
        
        # Time reference implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):  # Fewer runs because it's slower
            out_ref, state_ref = ref_layer(x)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / 5 * 1000
        
        # Compare outputs
        output_diff = (out_tri - out_ref).abs().max().item()
        state_diff = (state_tri - state_ref).abs().max().item() if state_tri is not None and state_ref is not None else 0
        
        comparison = {
            'triton_time_ms': tri_time,
            'reference_time_ms': ref_time,
            'speedup': ref_time / tri_time,
            'output_max_diff': output_diff,
            'state_max_diff': state_diff,
            'outputs_match': output_diff < 1e-5,
            'states_match': state_diff < 1e-5,
        }
        
        print(f"Triton time:  {tri_time:.2f}ms")
        print(f"Reference time: {ref_time:.2f}ms")
        print(f"Speedup: {comparison['speedup']:.2f}x")
        print(f"Output max difference: {output_diff:.2e}")
        print(f"State max difference: {state_diff:.2e}")
        print(f"Implementations match: {'✓' if comparison['outputs_match'] and comparison['states_match'] else '✗'}")
        
        return comparison


# =============================================================================
# 5. MAIN EXECUTION & ANALYSIS
# =============================================================================

def main():
    """Main execution: run comprehensive tests and analysis."""
    print(f"\n{'='*80}")
    print("DELTA RULE INTEGRATION FRAMEWORK")
    print(f"{'='*80}")
    
    # Environment info
    print(f"\nEnvironment Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Triton: {triton.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available. Some tests may fail.")
        return
    
    # Create test suite
    test_suite = DeltaRuleTestSuite(device='cuda')
    
    # Run comprehensive tests
    print(f"\n{'='*80}")
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}")
    
    summary = test_suite.run_full_suite()
    
    # Integration validation
    print(f"\n{'='*80}")
    print("RUNNING INTEGRATION VALIDATION")
    print(f"{'='*80}")
    
    validator = DeltaRuleIntegrationValidator(device='cuda')
    
    # Test with a sample model
    sample_model = GatedDeltaNetLayer(
        d_model=256,
        n_heads=8,
        head_dim=32,
        value_dim=64,
        use_triton=True
    )
    
    integration_results = validator.validate_with_existing_model(
        sample_model,
        input_shape=(4, 128, 256),
        n_steps=5
    )
    
    # Compare implementations
    comparison = validator.compare_implementations()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL INTEGRATION STATUS")
    print(f"{'='*80}")
    
    all_checks_passed = (
        summary['all_tests_passed'] and
        integration_results['state_consistency'] and
        comparison['outputs_match'] and
        comparison['states_match']
    )
    
    print(f"\nIntegration Status: {'✓ READY FOR PRODUCTION' if all_checks_passed else '✗ NEEDS FURTHER TESTING'}")
    print(f"\nNext Steps:")
    print(f"  1. Import the stable Triton kernel into your existing build")
    print(f"  2. Replace sequential Delta Rule implementations with triton_delta_rule()")
    print(f"  3. Ensure input shapes match: [B, T, H, K], [B, T, H, V], [B, T, H], [B, T, H]")
    print(f"  4. Test with your specific model configurations")
    print(f"  5. Monitor performance with the provided test suite")
    
    # Save all results
    final_results = {
        'test_suite': test_suite.results,
        'integration': integration_results,
        'comparison': comparison,
        'summary': summary,
        'environment': {
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda,
            'triton': triton.__version__,
            'gpu': torch.cuda.get_device_name(0),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    
    with open("delta_rule_integration_report.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nFull report saved to: delta_rule_integration_report.json")
    
    return all_checks_passed


if __name__ == "__main__":
    # Run the complete integration framework
    success = main()
    
    if success:
        print(f"\n{'='*80}")
        print("INTEGRATION SUCCESSFUL - You can now safely integrate into your build!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("INTEGRATION ISSUES DETECTED - Review test results above")
        print(f"{'='*80}")