"""
GroundThink v7 Model Components

- GatedDeltaNetLayer: TRUE Delta Rule with chunk-recurrent kernels
- SlidingWindowAttention: Local attention + state retrieval
- TransparentHybrid: Full model with GDN→SWA information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

# Use absolute imports for flat-folder/notebook compatibility
# For package usage, run from parent dir with: python -m groundthink_v7.script
try:
    from .config import HybridConfig
    from .core import chunk_delta_rule
except ImportError:
    from config import HybridConfig
    from core import chunk_delta_rule

__all__ = ['GatedDeltaNetLayer', 'SlidingWindowAttention', 'TransparentHybrid', 'RMSNorm', 'SwiGLUFFN']

# Check for flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for GDN keys.
    
    This makes keys position-dependent, preventing slot collision
    when the same token (e.g., MARKER) appears at different positions.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin
        self._cos_cache = None
        self._sin_cache = None
        self._cached_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
            self._cos_cache = emb.cos().to(dtype)
            self._sin_cache = emb.sin().to(dtype)
            self._cached_seq_len = seq_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: [B, T, H, K] tensor (keys)
        
        Returns:
            x with rotary embeddings applied
        """
        B, T, H, K = x.shape
        self._update_cache(T, x.device, x.dtype)
        
        cos = self._cos_cache[:T].view(1, T, 1, K)
        sin = self._sin_cache[:T].view(1, T, 1, K)
        
        # Rotate: split in half, swap, negate
        x1, x2 = x[..., :K//2], x[..., K//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, expansion: float = 8/3):
        super().__init__()
        hidden = ((int(d_model * expansion) + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GatedDeltaNetLayer(nn.Module):
    """
    GDN layer with orthogonal key bank for multi-needle retrieval.
    
    TRUE Delta Rule: S_t = g_t * S_{t-1} + β_t * (v_t - S_{t-1}·k_t) ⊗ k_t
    
    Orthogonal Key Bank:
        - Pre-allocate K orthogonal keys per head
        - MARKER tokens assigned slots via counter (sequential)
        - Guarantees zero interference between stored needles
    """
    def __init__(self, cfg: HybridConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
        
        # Standard key projection (for non-MARKER tokens)
        self.k_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        
        # ORTHOGONAL KEY BANK: [n_heads, bank_size, head_dim]
        # bank_size is capped at K since we can't have more than K orthogonal vectors in R^K
        self.bank_size = min(getattr(cfg, 'key_bank_size', K), K)
        key_bank = torch.zeros(H, self.bank_size, K)
        for h in range(H):
            # Create [bank_size, K] matrix, QR gives Q=[bank_size, bank_size] orthonormal rows
            random_matrix = torch.randn(self.bank_size, K)
            Q, _ = torch.linalg.qr(random_matrix.T)  # Q is [K, bank_size]
            key_bank[h] = Q.T  # Transpose to [bank_size, K]
        self.register_buffer('key_bank', key_bank)
        
        # Query projection for CUE tokens (maps CUE embedding → query over bank)
        self.cue_query_proj = nn.Linear(cfg.d_model, self.bank_size, bias=True)
        
        self.v_proj = nn.Linear(cfg.d_model, H * V, bias=False)
        self.o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        # Gates
        self.beta_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias)
        
        self.g_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.g_proj.bias, cfg.g_bias)
        
        self.norm = RMSNorm(cfg.d_model)
        self.use_shifted_value = getattr(cfg, 'shifted_value', True)
        
        self.marker_token = getattr(cfg, 'marker_token', 50251)
        self.cue_token = getattr(cfg, 'cue_token', 50250)
        
    def forward(
        self, 
        x: torch.Tensor, 
        initial_state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        MARKER tokens get assigned orthogonal keys from bank (counter-based).
        CUE tokens query the bank via learned projection.
        """
        B, T, D = x.shape
        H, K, V = self.cfg.n_heads, self.cfg.head_dim, self.cfg.value_dim
        
        x_norm = self.norm(x)
        
        # Standard keys for all positions
        k_full = self.k_proj(x_norm).view(B, T, H, K)
        
        # ORTHOGONAL KEY BANK: Assign sequential slots to MARKER tokens
        if input_ids is not None:
            # Vectorized: find all MARKER positions, assign sequential slots
            for b in range(B):
                marker_mask = input_ids[b] == self.marker_token
                if marker_mask.any():
                    marker_positions = marker_mask.nonzero().squeeze(-1)  # [num_markers]
                    num_markers = marker_positions.numel()
                    slot_indices = torch.arange(num_markers, device=x.device) % self.bank_size
                    # Gather bank keys for these slots: [num_markers, H, K]
                    bank_keys = self.key_bank[:, slot_indices, :].transpose(0, 1)  # [num_markers, H, K]
                    k_full[b, marker_positions] = bank_keys.to(k_full.dtype)
        
        v_full = self.v_proj(x_norm).view(B, T, H, V)
        
        # Soft sparsity factor: non-special tokens get reduced β, not zero
        beta_floor = getattr(self.cfg, 'beta_floor', 0.1)
        
        if self.use_shifted_value and T > 1:
            k = k_full[:, :-1]
            v = v_full[:, 1:]
            beta = torch.sigmoid(self.beta_proj(x_norm[:, :-1]))
            g = torch.sigmoid(self.g_proj(x_norm[:, :-1]))
            
            # SOFT SPARSITY: MARKER tokens get full β, others get reduced but not zero
            # This allows learning from normal text while still prioritizing markers
            if input_ids is not None:
                marker_mask = (input_ids[:, :-1] == self.marker_token).unsqueeze(-1)  # [B, T-1, 1]
                # marker_mask=True → 1.0, marker_mask=False → beta_floor
                beta_scale = beta_floor + (1.0 - beta_floor) * marker_mask.to(beta.dtype)
                beta = beta * beta_scale
        else:
            k = k_full
            v = v_full
            beta = torch.sigmoid(self.beta_proj(x_norm))
            g = torch.sigmoid(self.g_proj(x_norm))
            
            if input_ids is not None:
                marker_mask = (input_ids == self.marker_token).unsqueeze(-1)
                beta_scale = beta_floor + (1.0 - beta_floor) * marker_mask.to(beta.dtype)
                beta = beta * beta_scale
        
        # L2 normalize keys
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        
        # Initialize state - ALWAYS FP32 for SSM stability (per references)
        # Even under AMP, accumulating state needs higher precision
        if initial_state is None:
            state = torch.zeros(B, H, K, V, device=x.device, dtype=torch.float32)
        else:
            state = initial_state.to(torch.float32)
        
        # Chunk-recurrent Delta Rule (operates in FP32 for state)
        out, new_state = chunk_delta_rule(k, v, beta, g, state, self.cfg.chunk_size)
        
        if self.use_shifted_value and T > 1:
            zero_pad = torch.zeros(B, 1, H, V, device=x.device, dtype=out.dtype)
            out = torch.cat([zero_pad, out], dim=1)
        
        output = out.to(x.dtype).reshape(B, T, H * V)
        output = x + self.o_proj(output)
        
        # Diagnostics
        n_markers = 0
        if input_ids is not None:
            n_markers = (input_ids == self.marker_token).sum().item()
        
        diag = {
            'beta_mean': beta.mean().item(),
            'beta_max': beta.max().item(),
            'cached_keys': k_full,  # Cache the ACTUAL computed keys for SWA to use
            'g_mean': g.mean().item(),
            'state_norm': new_state.norm().item(),
            'n_markers': n_markers,
        }
        
        return output, new_state, diag


class SlidingWindowAttention(nn.Module):
    """
    SWA with state retrieval from GDN.
    
    Two pathways:
        1. Local attention (flash_attn or PyTorch fallback)
        2. State retrieval: queries GDN state for global context
    
    For multi-needle retrieval:
        - CUE tokens use orthogonal bank keys as queries
        - CUE_0 → bank slot 0, CUE_1 → bank slot 1, etc.
    """
    def __init__(self, cfg: HybridConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = K ** -0.5
        
        # Local attention
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        
        # State retrieval
        self.global_q_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        nn.init.normal_(self.global_q_proj.weight, std=cfg.init_std)
        self.retrieval_o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        # Retrieval gate
        self.gate_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.gate_proj.bias, 1.0)
        
        # CUE token detection
        self.cue_token = getattr(cfg, 'cue_token', 50250)
        self.bank_size = getattr(cfg, 'key_bank_size', K)
        
        # RoPE for local attention (position awareness)
        self.use_rope = getattr(cfg, 'use_rope', False)
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                dim=self.head_dim,
                base=getattr(cfg, 'rope_base', 10000.0)
            )
        
        # Bottleneck shortcut: permanent scale on local attention path
        # Always active (not like dropout which is off at eval)
        self.local_scale = getattr(cfg, 'local_scale', 0.3)
        
        # Stochastic local drop during training
        self.local_drop_prob = getattr(cfg, 'local_drop_prob', 0.7)
        
        self.norm = RMSNorm(cfg.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        gdn_state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        key_bank: Optional[torch.Tensor] = None,
        gdn_cached_keys: Optional[torch.Tensor] = None  # CACHED keys from GDN (computed on GDN's x)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, D] input
            gdn_state: [B, H, K, V] state from GDN layers
            input_ids: [B, T] token IDs for CUE detection
            key_bank: [H, bank_size, K] orthogonal keys from GDN
            gdn_cached_keys: [B, T, H, K] ACTUAL keys GDN computed (not a projection to apply)
            
        Returns:
            output: [B, T, D]
            diag: diagnostic dict
        """
        B, T, D = x.shape
        H = self.cfg.n_heads
        K, V, W = self.cfg.head_dim, self.cfg.value_dim, self.cfg.window_size
        
        x_norm = self.norm(x)
        
        # === Local Attention ===
        q = self.q_proj(x_norm).view(B, T, H, self.head_dim)
        k = self.k_proj(x_norm).view(B, T, H, self.head_dim)
        v = self.v_proj(x_norm).view(B, T, H, self.head_dim)
        
        # Apply RoPE to q/k for position-aware local attention
        if self.use_rope:
            q = self.rotary(q)
            k = self.rotary(k)
        
        if FLASH_ATTN_AVAILABLE:
            local_out = flash_attn_func(q, k, v, causal=True, window_size=(W, 0))
            local_out = local_out.reshape(B, T, D)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            mask |= torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-W - 1)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            local_out = (F.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, T, D)
        
        local_out = self.o_proj(local_out)
        # Stochastic local drop: during training, randomly drop local attention
        # to force model to learn state retrieval pathway
        if self.training:
            # Stochastic drop: 70% probability of dropping local output
            drop_prob = getattr(self, 'local_drop_prob', 0.7)
            keep_mask = (torch.rand(B, 1, 1, device=x.device) > drop_prob).to(local_out.dtype)
            # Scale up by 1/(1-drop_prob) to maintain expected value
            local_out = local_out * keep_mask / (1.0 - drop_prob)
        else:
            # Inference: scale down to match expected training magnitude
            local_out = self.local_scale * local_out
        
        # === State Retrieval ===
        retrieval_out = torch.zeros_like(x)
        gate_mean = 0.0
        n_cue_queries = 0
        
        if gdn_state is not None:
            # Use CACHED keys from GDN directly - these are the actual keys GDN wrote with
            if gdn_cached_keys is not None:
                q_g = gdn_cached_keys  # [B, T, H, K] - EXACT keys GDN used, perfectly aligned
            else:
                q_g = self.global_q_proj(x_norm).view(B, T, H, K)  # Fallback (won't align)
            
            # For CUE tokens, use orthogonal bank keys as queries
            # CUE_0 (token 250) → slot 0, CUE_1 (251) → slot 1, etc.
            if input_ids is not None and key_bank is not None:
                for b in range(B):
                    for cue_idx in range(self.bank_size):
                        cue_token_id = self.cue_token + cue_idx
                        cue_mask = input_ids[b] == cue_token_id
                        if cue_mask.any():
                            cue_positions = cue_mask.nonzero().squeeze(-1)
                            # Use bank key for this slot as query
                            bank_key = key_bank[:, cue_idx, :]  # [H, K]
                            q_g[b, cue_positions] = bank_key.to(q_g.dtype)
                            n_cue_queries += cue_positions.numel()
            
            q_g = q_g.transpose(1, 2)  # [B, H, T, K]
            # NOTE: No ReLU for bank key queries - they need both +/- components
            
            retrieved = torch.einsum('bhkv,bhtk->bhtv', gdn_state.to(x.dtype), q_g)
            retrieved = retrieved.transpose(1, 2).reshape(B, T, H * V)
            retrieval_out = self.retrieval_o_proj(retrieved)
            
            gate = torch.sigmoid(self.gate_proj(x_norm))
            gate_mean = gate.mean().item()
            retrieval_out = gate.mean(dim=-1, keepdim=True) * retrieval_out
        
        out = x + local_out + retrieval_out
        
        diag = {
            'gate_mean': gate_mean,
            'local_norm': local_out.norm().item(),
            'retrieval_norm': retrieval_out.norm().item(),
            'n_cue_queries': n_cue_queries,
        }
        
        return out, diag


class ParallelHybridLayer(nn.Module):
    """
    Parallel GDN + SWA in SAME layer (Hymba-style fusion).
    
    Both GDN and SWA see the SAME input x, so shared_key_proj(x) produces
    identical keys for writes and queries. This guarantees alignment.
    
    Information Flow:
        x_norm = norm(x)
        gdn_out, state = GDN(x_norm)  # Updates state with k=shared_key_proj(x_norm)
        swa_out = SWA(x_norm, state)   # Queries with q=shared_key_proj(x_norm)
        out = x + gdn_out + swa_out    # Residual combines both
    """
    def __init__(self, cfg: HybridConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
        
        # SHARED key projection - GDN writes and SWA queries use this on SAME x
        self.shared_key_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        
        # GDN components (value, gates, output)
        self.v_proj = nn.Linear(cfg.d_model, H * V, bias=False)
        self.gdn_o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        self.beta_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias)
        
        self.g_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.g_proj.bias, cfg.g_bias)
        
        # SWA local attention (separate projections - not for state retrieval)
        self.local_q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.local_k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.local_v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.swa_o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        
        # State retrieval output
        self.retrieval_o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        # Retrieval gate
        self.gate_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.gate_proj.bias, 1.0)
        
        self.norm = RMSNorm(cfg.d_model)
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = K ** -0.5
        
        self.use_shifted_value = getattr(cfg, 'shifted_value', True)
        self.marker_token = getattr(cfg, 'marker_token', 50251)
        self.local_drop_prob = getattr(cfg, 'local_drop_prob', 0.7)
        self.local_scale = getattr(cfg, 'local_scale', 0.3)
        
    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Parallel GDN + SWA on SAME x.
        
        Returns:
            output: [B, T, D]
            new_state: [B, H, K, V]
            diag: diagnostic dict
        """
        B, T, D = x.shape
        H, K, V = self.cfg.n_heads, self.cfg.head_dim, self.cfg.value_dim
        W = self.cfg.window_size
        
        x_norm = self.norm(x)
        
        # === SHARED KEY PROJECTION - same for GDN write and SWA query ===
        k_shared = self.shared_key_proj(x_norm).view(B, T, H, K)
        k_shared = F.normalize(k_shared.float(), p=2, dim=-1).to(x.dtype)
        
        # === GDN: Write to state ===
        v_full = self.v_proj(x_norm).view(B, T, H, V)
        
        beta_floor = getattr(self.cfg, 'beta_floor', 0.1)
        
        if self.use_shifted_value and T > 1:
            k = k_shared[:, :-1]
            v = v_full[:, 1:]
            beta = torch.sigmoid(self.beta_proj(x_norm[:, :-1]))
            g = torch.sigmoid(self.g_proj(x_norm[:, :-1]))
            
            if input_ids is not None:
                marker_mask = (input_ids[:, :-1] == self.marker_token).unsqueeze(-1)
                beta_scale = beta_floor + (1.0 - beta_floor) * marker_mask.to(beta.dtype)
                beta = beta * beta_scale
        else:
            k = k_shared
            v = v_full
            beta = torch.sigmoid(self.beta_proj(x_norm))
            g = torch.sigmoid(self.g_proj(x_norm))
            
            if input_ids is not None:
                marker_mask = (input_ids == self.marker_token).unsqueeze(-1)
                beta_scale = beta_floor + (1.0 - beta_floor) * marker_mask.to(beta.dtype)
                beta = beta * beta_scale
        
        if initial_state is None:
            state = torch.zeros(B, H, K, V, device=x.device, dtype=x.dtype)
        else:
            state = initial_state.to(x.dtype)
        
        gdn_out, new_state = chunk_delta_rule(k, v, beta, g, state, self.cfg.chunk_size)
        
        if self.use_shifted_value and T > 1:
            zero_pad = torch.zeros(B, 1, H, V, device=x.device, dtype=gdn_out.dtype)
            gdn_out = torch.cat([zero_pad, gdn_out], dim=1)
        
        gdn_out = gdn_out.to(x.dtype).reshape(B, T, H * V)
        gdn_out = self.gdn_o_proj(gdn_out)
        
        # === SWA: Local attention ===
        q = self.local_q_proj(x_norm).view(B, T, H, self.head_dim)
        k_local = self.local_k_proj(x_norm).view(B, T, H, self.head_dim)
        v_local = self.local_v_proj(x_norm).view(B, T, H, self.head_dim)
        
        if FLASH_ATTN_AVAILABLE:
            local_out = flash_attn_func(q, k_local, v_local, causal=True, window_size=(W, 0))
            local_out = local_out.reshape(B, T, D)
        else:
            q = q.transpose(1, 2)
            k_local = k_local.transpose(1, 2)
            v_local = v_local.transpose(1, 2)
            
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            mask |= torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-W - 1)
            
            attn = (q @ k_local.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            local_out = (F.softmax(attn, dim=-1) @ v_local).transpose(1, 2).reshape(B, T, D)
        
        local_out = self.swa_o_proj(local_out)
        
        if self.training:
            keep_mask = (torch.rand(B, 1, 1, device=x.device) > self.local_drop_prob).to(local_out.dtype)
            local_out = local_out * keep_mask / (1.0 - self.local_drop_prob)
        else:
            local_out = self.local_scale * local_out
        
        # === SWA: State retrieval using SAME shared keys as query ===
        q_g = k_shared.transpose(1, 2)  # [B, H, T, K] - EXACT same as GDN write keys!
        retrieved = torch.einsum('bhkv,bhtk->bhtv', new_state.to(x.dtype), q_g)
        retrieved = retrieved.transpose(1, 2).reshape(B, T, H * V)
        retrieval_out = self.retrieval_o_proj(retrieved)
        
        gate = torch.sigmoid(self.gate_proj(x_norm))
        retrieval_out = gate.mean(dim=-1, keepdim=True) * retrieval_out
        
        # === Combine: residual + GDN + local + retrieval ===
        out = x + gdn_out + local_out + retrieval_out
        
        diag = {
            'beta_mean': beta.mean().item(),
            'g_mean': g.mean().item(),
            'state_norm': new_state.norm().item(),
            'gate_mean': gate.mean().item(),
            'local_norm': local_out.norm().item(),
            'retrieval_norm': retrieval_out.norm().item(),
            'gdn_norm': gdn_out.norm().item(),
            'layer': 'P',  # Parallel
            'layer_idx': self.layer_idx,
        }
        
        return out, new_state, diag


class TransparentHybrid(nn.Module):
    """
    GDN + SWA hybrid model.
    
    Information Flow:
        - GDN layers compress sequence into state S_t [H, K, V]
        - State flows to subsequent SWA layers for retrieval
        - SWA provides precision retrieval (window + global via state)
    
    Layer pattern examples:
        "GS"       - 2 layers: GDN, SWA (stacked - DEPRECATED, keys misalign)
        "GGS"      - 3 layers: 2 GDN, 1 SWA  
        "P"        - 1 layer: Parallel GDN+SWA (RECOMMENDED)
        "PP"       - 2 layers: 2x Parallel
    """
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        self.embed_norm = RMSNorm(cfg.d_model)
        
        self.layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        
        for i, lt in enumerate(cfg.layer_pattern):
            if lt == 'G':
                self.layers.append(GatedDeltaNetLayer(cfg, i))
            elif lt == 'S':
                self.layers.append(SlidingWindowAttention(cfg, i))
            elif lt == 'P':
                self.layers.append(ParallelHybridLayer(cfg, i))
            else:
                raise ValueError(f"Unknown layer type: {lt}")
            self.ffns.append(SwiGLUFFN(cfg.d_model))
        
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token indices
            targets: [B, T] optional target indices
            return_diagnostics: whether to return layer diags
            
        Returns:
            logits: [B, T, vocab_size]
            loss: scalar if targets provided
            all_diag: list of layer diagnostics
            state: final GDN state
        """
        x = self.embed(input_ids)
        x = self.embed_norm(x)
        state = None
        key_bank = None  # Will be set by first GDN layer
        all_diag = []
        
        gdn_cached_keys = None  # For legacy stacked mode
        for i, (layer, ffn) in enumerate(zip(self.layers, self.ffns)):
            lt = self.cfg.layer_pattern[i]
            if lt == 'P':
                # Parallel: GDN+SWA in same layer, both see same x
                x, state, diag = layer(x, initial_state=state, input_ids=input_ids)
            elif lt == 'G':
                # Pass input_ids for MARKER detection and orthogonal key bank
                x, state, diag = layer(x, initial_state=state, input_ids=input_ids)
                key_bank = layer.key_bank  # Get bank for SWA to use
                if 'cached_keys' in diag:
                    gdn_cached_keys = diag.pop('cached_keys')  # Legacy
            else:  # 'S'
                # Pass input_ids, key_bank, AND cached keys for aligned retrieval
                x, diag = layer(x, gdn_state=state, input_ids=input_ids, key_bank=key_bank, gdn_cached_keys=gdn_cached_keys)
            x = ffn(x)
            if 'layer' not in diag:
                diag['layer'] = lt
            if 'layer_idx' not in diag:
                diag['layer_idx'] = i
            all_diag.append(diag)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss, all_diag, state
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
