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

from .config import HybridConfig
from .core import chunk_delta_rule

__all__ = ['GatedDeltaNetLayer', 'SlidingWindowAttention', 'TransparentHybrid', 'RMSNorm', 'SwiGLUFFN']

# Check for flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


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
    GDN layer with chunk-recurrent Triton kernels.
    
    TRUE Delta Rule: S_t = g_t * S_{t-1} + β_t * (v_t - S_{t-1}·k_t) ⊗ k_t
    
    The chunk-recurrent implementation ensures numerical stability
    by checkpointing states at chunk boundaries (no division chains).
    """
    def __init__(self, cfg: HybridConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
        
        # Projections
        self.k_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, H * V, bias=False)
        self.o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        # Gates with biased initialization
        self.beta_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.beta_proj.bias, cfg.beta_bias)
        
        self.g_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.g_proj.bias, cfg.g_bias)
        
        self.norm = RMSNorm(cfg.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, D] input
            initial_state: [B, H, K, V] optional
            
        Returns:
            output: [B, T, D] with residual
            state: [B, H, K, V] final state
            diag: diagnostic dict
        """
        B, T, D = x.shape
        H, K, V = self.cfg.n_heads, self.cfg.head_dim, self.cfg.value_dim
        
        x_norm = self.norm(x)
        
        # Projections
        k = self.k_proj(x_norm).view(B, T, H, K)
        v = self.v_proj(x_norm).view(B, T, H, V)
        
        # L2 normalize keys (CRITICAL for Delta Rule)
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        
        # Gates
        beta = torch.sigmoid(self.beta_proj(x_norm))
        g = torch.sigmoid(self.g_proj(x_norm))
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(B, H, K, V, device=x.device, dtype=x.dtype)
        else:
            state = initial_state.to(x.dtype)
        
        # Chunk-recurrent Delta Rule (stable!)
        out, new_state = chunk_delta_rule(k, v, beta, g, state, self.cfg.chunk_size)
        
        # Output projection + residual
        output = out.to(x.dtype).reshape(B, T, H * V)
        output = x + self.o_proj(output)
        
        diag = {
            'beta_mean': beta.mean().item(),
            'beta_max': beta.max().item(),
            'g_mean': g.mean().item(),
            'state_norm': new_state.norm().item(),
            'state_max': new_state.abs().max().item(),
        }
        
        return output, new_state, diag


class SlidingWindowAttention(nn.Module):
    """
    SWA with state retrieval from GDN.
    
    Two pathways:
        1. Local attention (flash_attn or PyTorch fallback)
        2. State retrieval: queries GDN state for global context
    
    This enables NIAH - SWA can "see" information stored in GDN state
    from anywhere in the sequence.
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
        
        self.norm = RMSNorm(cfg.d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        gdn_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, T, D] input
            gdn_state: [B, H, K, V] state from GDN layers
            
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
        
        # === State Retrieval ===
        retrieval_out = torch.zeros_like(x)
        gate_mean = 0.0
        
        if gdn_state is not None:
            q_g = self.global_q_proj(x_norm).view(B, T, H, K).transpose(1, 2)
            q_g = F.relu(q_g)  # Sparse queries
            
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
        }
        
        return out, diag


class TransparentHybrid(nn.Module):
    """
    GDN + SWA hybrid model.
    
    Information Flow:
        - GDN layers compress sequence into state S_t [H, K, V]
        - State flows to subsequent SWA layers for retrieval
        - SWA provides precision retrieval (window + global via state)
    
    Layer pattern examples:
        "GS"       - 2 layers: GDN, SWA
        "GGS"      - 3 layers: 2 GDN, 1 SWA  
        "GGSG"     - 4 layers: GDN, GDN, SWA, GDN
    """
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        
        self.layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        
        for i, lt in enumerate(cfg.layer_pattern):
            if lt == 'G':
                self.layers.append(GatedDeltaNetLayer(cfg, i))
            elif lt == 'S':
                self.layers.append(SlidingWindowAttention(cfg, i))
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
        state = None
        all_diag = []
        
        for i, (layer, ffn) in enumerate(zip(self.layers, self.ffns)):
            lt = self.cfg.layer_pattern[i]
            if lt == 'G':
                x, state, diag = layer(x, initial_state=state)
            else:
                x, diag = layer(x, gdn_state=state)
            x = ffn(x)
            diag['layer'] = lt
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
