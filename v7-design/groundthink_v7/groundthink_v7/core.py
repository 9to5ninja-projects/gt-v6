"""
GroundThink v7 Core - Chunk-Recurrent Delta Rule

This module contains:
- Triton kernels for TRUE Delta Rule (forward + backward)
- Chunk-recurrent algorithm for numerical stability
- Autograd integration

The key insight: instead of reconstructing states via unstable division,
we checkpoint at chunk boundaries and recompute forward (stable).

Usage:
    from groundthink_v7.core import chunk_delta_rule
    output, final_state = chunk_delta_rule(k, v, beta, g, initial_state)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple

__all__ = ['chunk_delta_rule', 'CHUNK_SIZE']

CHUNK_SIZE = 64  # Default chunk size


# =============================================================================
# FORWARD KERNEL
# =============================================================================

@triton.jit
def _chunk_fwd_kernel(
    K, V, Beta, G, State_in, Out, Checkpoints,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    stride_c_b, stride_c_n, stride_c_h, stride_c_k, stride_c_v,
    T: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr, CHUNK: tl.constexpr,
):
    """Forward pass with checkpoint at each chunk boundary."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    # Load initial state
    s_base = State_in + pid_b * stride_s_b + pid_h * stride_s_h
    state = tl.load(s_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v).to(tl.float32)
    
    # Checkpoint initial state
    c_base = Checkpoints + pid_b * stride_c_b + pid_h * stride_c_h
    tl.store(c_base + k_offs[:, None] * stride_c_k + v_offs[None, :] * stride_c_v, state)
    
    n_chunks = tl.cdiv(T, CHUNK)
    
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * CHUNK
        chunk_end = tl.minimum(chunk_start + CHUNK, T)
        
        for t in range(chunk_start, chunk_end):
            # Load inputs
            k_base = K + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
            k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
            
            v_base = V + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
            v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
            
            bg_off = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
            beta_t = tl.load(Beta + bg_off).to(tl.float32)
            g_t = tl.load(G + bg_off).to(tl.float32)
            
            # TRUE Delta Rule
            pred = tl.sum(state * k_t[:, None], axis=0)
            error = v_t - pred
            outer = k_t[:, None] * error[None, :]
            state = g_t * state + beta_t * outer
            
            # Output
            out_t = tl.sum(state * k_t[:, None], axis=0)
            o_base = Out + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
            tl.store(o_base + v_offs * stride_o_v, out_t)
        
        # Checkpoint after chunk
        c_ptrs = c_base + (chunk_idx + 1) * stride_c_n + k_offs[:, None] * stride_c_k + v_offs[None, :] * stride_c_v
        tl.store(c_ptrs, state)


# =============================================================================
# BACKWARD KERNEL (Optimized - O(T) not O(T×C))
# =============================================================================

@triton.jit
def _chunk_bwd_kernel(
    K, V, Beta, G, Checkpoints,
    dOut, dState_out,
    dK, dV, dBeta, dG, dState_in,
    ChunkStates,  # Temp buffer [B, H, CHUNK, K, V]
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_c_b, stride_c_n, stride_c_h, stride_c_k, stride_c_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    stride_dk_b, stride_dk_t, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_t, stride_dv_h, stride_dv_v,
    stride_dbg_b, stride_dbg_t, stride_dbg_h,
    stride_ds_b, stride_ds_h, stride_ds_k, stride_ds_v,
    stride_cs_b, stride_cs_h, stride_cs_t, stride_cs_k, stride_cs_v,
    T: tl.constexpr, K_DIM: tl.constexpr, V_DIM: tl.constexpr, CHUNK: tl.constexpr,
):
    """
    Backward pass using checkpoints + intra-chunk state caching.
    
    For each chunk:
      1. Load checkpoint, forward through chunk saving states → O(C)
      2. Backward through chunk using saved states → O(C)
    Total: O(T) with no division chains.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    c_base = Checkpoints + pid_b * stride_c_b + pid_h * stride_c_h
    cs_base = ChunkStates + pid_b * stride_cs_b + pid_h * stride_cs_h
    n_chunks = tl.cdiv(T, CHUNK)
    
    # Load upstream dstate
    ds_base = dState_out + pid_b * stride_ds_b + pid_h * stride_ds_h
    dstate = tl.load(ds_base + k_offs[:, None] * stride_ds_k + v_offs[None, :] * stride_ds_v).to(tl.float32)
    
    for chunk_rev in range(n_chunks):
        chunk_idx = n_chunks - 1 - chunk_rev
        chunk_start = chunk_idx * CHUNK
        chunk_end = tl.minimum(chunk_start + CHUNK, T)
        chunk_len = chunk_end - chunk_start
        
        # === Phase 1: Forward through chunk, cache states ===
        ckpt_ptrs = c_base + chunk_idx * stride_c_n + k_offs[:, None] * stride_c_k + v_offs[None, :] * stride_c_v
        state = tl.load(ckpt_ptrs).to(tl.float32)
        
        for local_t in range(chunk_len):
            t = chunk_start + local_t
            
            # Cache S_{t-1}
            cs_ptrs = cs_base + local_t * stride_cs_t + k_offs[:, None] * stride_cs_k + v_offs[None, :] * stride_cs_v
            tl.store(cs_ptrs, state)
            
            # Forward step
            k_base = K + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
            k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
            v_base = V + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
            v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
            bg_off = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
            beta_t = tl.load(Beta + bg_off).to(tl.float32)
            g_t = tl.load(G + bg_off).to(tl.float32)
            
            pred = tl.sum(state * k_t[:, None], axis=0)
            error = v_t - pred
            outer = k_t[:, None] * error[None, :]
            state = g_t * state + beta_t * outer
        
        # === Phase 2: Backward through chunk ===
        for local_rev in range(chunk_len):
            local_t = chunk_len - 1 - local_rev
            t = chunk_start + local_t
            
            # Load cached S_{t-1}
            cs_ptrs = cs_base + local_t * stride_cs_t + k_offs[:, None] * stride_cs_k + v_offs[None, :] * stride_cs_v
            state_prev = tl.load(cs_ptrs).to(tl.float32)
            
            # Load inputs
            k_base = K + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
            k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
            v_base = V + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
            v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
            bg_off = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
            beta_t = tl.load(Beta + bg_off).to(tl.float32)
            g_t = tl.load(G + bg_off).to(tl.float32)
            
            # Recompute S_t
            pred_t = tl.sum(state_prev * k_t[:, None], axis=0)
            error_t = v_t - pred_t
            outer_t = k_t[:, None] * error_t[None, :]
            state_t = g_t * state_prev + beta_t * outer_t
            
            # Load dout
            dout_base = dOut + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
            dout_t = tl.load(dout_base + v_offs * stride_o_v).to(tl.float32)
            
            # Accumulate dstate from output
            dstate = dstate + k_t[:, None] * dout_t[None, :]
            
            # Compute gradients
            dstate_dot_k = tl.sum(dstate * k_t[:, None], axis=0)
            
            dv_t = beta_t * dstate_dot_k
            dbeta_t = tl.sum(dstate * outer_t)
            dg_t = tl.sum(dstate * state_prev)
            
            dk_out = tl.sum(state_t * dout_t[None, :], axis=1)
            dk_outer = beta_t * tl.sum(dstate * error_t[None, :], axis=1)
            dk_pred = -beta_t * tl.sum(state_prev * dstate_dot_k[None, :], axis=1)
            dk_t = dk_out + dk_outer + dk_pred
            
            # Store
            dk_base = dK + pid_b * stride_dk_b + t * stride_dk_t + pid_h * stride_dk_h
            tl.store(dk_base + k_offs * stride_dk_d, dk_t)
            dv_base = dV + pid_b * stride_dv_b + t * stride_dv_t + pid_h * stride_dv_h
            tl.store(dv_base + v_offs * stride_dv_v, dv_t)
            dbeta_base = dBeta + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
            tl.store(dbeta_base, dbeta_t)
            dg_base = dG + pid_b * stride_dbg_b + t * stride_dbg_t + pid_h * stride_dbg_h
            tl.store(dg_base, dg_t)
            
            # Propagate dstate
            dstate = g_t * dstate - beta_t * k_t[:, None] * dstate_dot_k[None, :]
    
    # Store dstate_in
    ds_in_base = dState_in + pid_b * stride_ds_b + pid_h * stride_ds_h
    tl.store(ds_in_base + k_offs[:, None] * stride_ds_k + v_offs[None, :] * stride_ds_v, dstate)


# =============================================================================
# PYTHON WRAPPERS
# =============================================================================

def _forward(k, v, beta, g, initial_state, chunk_size):
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    dtype = k.dtype
    
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    
    if initial_state is None:
        state_in = torch.zeros(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    else:
        state_in = initial_state.contiguous().float()
    
    out = torch.empty(B, T, H, V_DIM, device=device, dtype=torch.float32)
    n_chunks = (T + chunk_size - 1) // chunk_size
    checkpoints = torch.empty(B, n_chunks + 1, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    
    _chunk_fwd_kernel[(B, H)](
        k, v, beta, g, state_in, out, checkpoints,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        state_in.stride(0), state_in.stride(1), state_in.stride(2), state_in.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        checkpoints.stride(0), checkpoints.stride(1), checkpoints.stride(2),
        checkpoints.stride(3), checkpoints.stride(4),
        T, K_DIM, V_DIM, chunk_size,
    )
    
    return out.to(dtype), checkpoints[:, -1].to(dtype), checkpoints


def _backward(k, v, beta, g, checkpoints, d_out, d_state_out, chunk_size):
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    checkpoints = checkpoints.contiguous().float()
    d_out = d_out.contiguous().float()
    d_state_out = d_state_out.contiguous().float()
    
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    d_beta = torch.empty_like(beta)
    d_g = torch.empty_like(g)
    d_state_in = torch.empty(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    chunk_states = torch.empty(B, H, chunk_size, K_DIM, V_DIM, device=device, dtype=torch.float32)
    
    _chunk_bwd_kernel[(B, H)](
        k, v, beta, g, checkpoints,
        d_out, d_state_out,
        d_k, d_v, d_beta, d_g, d_state_in,
        chunk_states,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        checkpoints.stride(0), checkpoints.stride(1), checkpoints.stride(2),
        checkpoints.stride(3), checkpoints.stride(4),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        d_k.stride(0), d_k.stride(1), d_k.stride(2), d_k.stride(3),
        d_v.stride(0), d_v.stride(1), d_v.stride(2), d_v.stride(3),
        d_beta.stride(0), d_beta.stride(1), d_beta.stride(2),
        d_state_in.stride(0), d_state_in.stride(1), d_state_in.stride(2), d_state_in.stride(3),
        chunk_states.stride(0), chunk_states.stride(1), chunk_states.stride(2),
        chunk_states.stride(3), chunk_states.stride(4),
        T, K_DIM, V_DIM, chunk_size,
    )
    
    return d_k, d_v, d_beta, d_g, d_state_in


class _ChunkDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, beta, g, initial_state, chunk_size):
        out, final_state, checkpoints = _forward(k, v, beta, g, initial_state, chunk_size)
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(k, v, beta, g, checkpoints)
        return out, final_state
    
    @staticmethod
    def backward(ctx, d_out, d_final_state):
        k, v, beta, g, checkpoints = ctx.saved_tensors
        d_k, d_v, d_beta, d_g, d_state = _backward(
            k, v, beta, g, checkpoints, d_out, d_final_state, ctx.chunk_size
        )
        return d_k, d_v, d_beta, d_g, d_state, None


def chunk_delta_rule(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk-recurrent TRUE Delta Rule.
    
    Numerically stable via gradient checkpointing - no division chains.
    
    Args:
        k: [B, T, H, K] - L2-normalized keys
        v: [B, T, H, V] - values
        beta: [B, T, H] - write gates (sigmoid output)
        g: [B, T, H] - forget gates (sigmoid output)
        initial_state: [B, H, K, V] - optional initial state
        chunk_size: checkpoint interval (default 64)
    
    Returns:
        output: [B, T, H, V]
        final_state: [B, H, K, V]
    """
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    
    if initial_state is None:
        initial_state = torch.zeros(B, H, K_DIM, V_DIM, device=k.device, dtype=k.dtype)
    
    return _ChunkDeltaFunction.apply(k, v, beta, g, initial_state, chunk_size)
