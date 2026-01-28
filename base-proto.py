import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet, FusedAttention # FusedAttention is efficient full attention
from fla.layers import SWAttention  # Or SWAttention for Sliding Window

class HybridFLAModel(nn.Module):
    def __init__(self, dim=512, num_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        
        # Create a 3:1 hybrid pattern (e.g., [G, G, G, A, G, G, G, A, G, G, G, A])
        for i in range(num_layers):
            # Attention layers at positions 3, 7, 11 (0-indexed)
            if i in [3, 7, 11]:
                # Use Sliding Window Attention (SWA) with window=512 for global context
                self.layers.append(SWAttention(hidden_size=dim, num_heads=8, window_size=512))
                # Alternatively, use FusedAttention for full attention:
                # self.layers.append(FusedAttention(hidden_size=dim, num_heads=8))
            else:
                # Use Gated DeltaNet for linear-complexity layers
                self.layers.append(GatedDeltaNet(hidden_size=dim))
        
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits