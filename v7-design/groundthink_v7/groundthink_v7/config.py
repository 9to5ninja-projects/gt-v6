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
