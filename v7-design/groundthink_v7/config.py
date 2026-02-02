"""
GroundThink v7 Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HybridConfig:
    """
    Configuration for TransparentHybrid model.
    
    Layer Pattern:
        String of 'G' (GDN) and 'S' (SWA), e.g. "GS", "GGS", "GGSG"
    
    State Matrix:
        Shape: [B, H, head_dim, value_dim]
        Capacity: head_dim × value_dim floats per head
    
    Gate Initialization:
        beta_bias < 0: sparse writes (gatekeeper)
        g_bias > 0: high retention
    """
    # Dimensions
    d_model: int = 256
    n_heads: int = 8
    head_dim: int = 32          # K dimension
    value_dim: int = 64         # V dimension (typically 2× head_dim)
    vocab_size: int = 50257
    
    # Architecture
    layer_pattern: str = "GS"
    window_size: int = 64       # SWA window
    chunk_size: int = 64        # Chunk size for gradient checkpointing
    
    # Initialization
    init_std: float = 0.02
    beta_bias: float = -2.0     # sigmoid(-2) ≈ 0.12
    g_bias: float = 2.0         # sigmoid(+2) ≈ 0.88
    
    # Soft sparsity: non-MARKER tokens get beta_floor * β, MARKER tokens get full β
    # 0.0 = binary (old broken mode), 0.1 = soft (recommended)
    beta_floor: float = 0.1
    
    # Shifted Value Mode (CRITICAL for NIAH)
    # When True: store (k_t, v_{t+1}) so state holds "what comes after" each key
    # When False: store (k_t, v_t) - original mode (broken for NIAH)
    shifted_value: bool = True
    
    # Orthogonal Key Bank (CRITICAL for multi-needle)
    # Pre-allocated orthogonal keys that MARKER tokens select from
    # This guarantees zero interference between stored needles
    key_bank_size: int = 64     # Max distinct keys per head (should be >= n_needles)
    
    # RoPE for positional encoding in SWA (local attention position awareness)
    use_rope: bool = False
    rope_base: float = 10000.0
    
    # Bottleneck shortcut: scale down SWA local attention path
    # Forces model to use state retrieval instead of bypassing via local window
    # 0.3 = local path contributes 30%, state path can dominate
    local_scale: float = 0.3
    
    # Stochastic local drop: probability of dropping local attention during training
    # Per practical_hybrid_solutions.md: 70% drop forces state retrieval learning
    local_drop_prob: float = 0.7
    
    # Special tokens (for NIAH testing)
    marker_token: int = 50251
    cue_token: int = 50250
    
    @property
    def n_layers(self) -> int:
        return len(self.layer_pattern)
    
    @property
    def n_gdn_layers(self) -> int:
        return sum(1 for c in self.layer_pattern if c == 'G')
    
    @property
    def n_swa_layers(self) -> int:
        return sum(1 for c in self.layer_pattern if c == 'S')
    
    @property
    def state_size_per_head(self) -> int:
        """Floats per head in state matrix."""
        return self.head_dim * self.value_dim
    
    def __repr__(self) -> str:
        return (f"HybridConfig({self.layer_pattern}, d={self.d_model}, "
                f"h={self.n_heads}, K={self.head_dim}, V={self.value_dim})")
