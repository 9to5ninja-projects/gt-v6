# =============================================================================
# CELL 2: GatedDeltaNetLayer - TRUE DELTA RULE (PURE PYTORCH)
# =============================================================================
#
# The FLA kernel `chunk_gated_delta_rule` processes chunks in parallel,
# which breaks external error correction. This implementation does true
# token-by-token Delta Rule updates.
#
# Delta Rule: S_t = S_{t-1} + β_t * (v_t - S_{t-1}·k_t) ⊗ k_t
#
# For efficiency, we still batch across B and H, just sequential over T.
# =============================================================================

class GatedDeltaNetLayer(nn.Module):
    """
    True Delta Rule GDN with pure PyTorch implementation.
    
    Key insight: Delta Rule MUST be applied token-by-token because
    each update depends on the current state for error correction.
    The FLA chunked kernel doesn't support this properly.
    """
    
    def __init__(self, cfg: HybridConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
        
        # Projections
        self.q_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, H * K, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, H * V, bias=False)
        self.o_proj = nn.Linear(H * V, cfg.d_model, bias=False)
        
        # Gates with gatekeeper initialization
        self.beta_proj = nn.Linear(cfg.d_model, H, bias=True)
        nn.init.constant_(self.beta_proj.bias, -2.0)  # Low default β
        
        # Optional: forget gate for gradual decay
        self.use_forget_gate = True
        if self.use_forget_gate:
            self.g_proj = nn.Linear(cfg.d_model, H, bias=True)
            nn.init.constant_(self.g_proj.bias, 2.0)  # High default retention
        
        self.norm = RMSNorm(cfg.d_model)
        self.scale = K ** -0.5
        
    def _delta_rule_step(
        self,
        state: torch.Tensor,  # [B, H, K, V]
        k_t: torch.Tensor,    # [B, H, K]
        v_t: torch.Tensor,    # [B, H, V]
        beta_t: torch.Tensor, # [B, H]
        g_t: torch.Tensor = None,  # [B, H] optional forget gate
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of true Delta Rule.
        
        S_new = g * S_old + β * (v - S_old·k) ⊗ k
        
        Returns: (output, new_state)
        """
        # Prediction from current state: [B, H, K, V] @ [B, H, K] -> [B, H, V]
        prediction = torch.einsum('bhkv,bhk->bhv', state, k_t)
        
        # Error: what we want to store minus what we'd retrieve
        error = v_t - prediction  # [B, H, V]
        
        # Outer product update: [B, H, V] ⊗ [B, H, K] -> [B, H, K, V]
        update = torch.einsum('bhv,bhk->bhkv', error, k_t)
        
        # Scale by beta: [B, H, 1, 1] * [B, H, K, V]
        update = beta_t.unsqueeze(-1).unsqueeze(-1) * update
        
        # Apply forget gate if used
        if g_t is not None:
            state = g_t.unsqueeze(-1).unsqueeze(-1) * state
        
        # New state
        new_state = state + update
        
        # Output: retrieve from new state using k as query
        # This gives the model access to just-stored information
        output = torch.einsum('bhkv,bhk->bhv', new_state, k_t)
        
        return output, new_state
    
    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        B, T, D = x.shape
        H, K, V = self.cfg.n_heads, self.cfg.head_dim, self.cfg.value_dim
        
        x_norm = self.norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(B, T, H, K)
        k = self.k_proj(x_norm).view(B, T, H, K)
        v = self.v_proj(x_norm).view(B, T, H, V)
        
        # CRITICAL: Normalize keys for Delta Rule stability
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        
        # Gates
        beta = torch.sigmoid(self.beta_proj(x_norm))  # [B, T, H]
        if self.use_forget_gate:
            g = torch.sigmoid(self.g_proj(x_norm))  # [B, T, H]
        else:
            g = None
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(B, H, K, V, device=x.device, dtype=x.dtype)
        else:
            state = initial_state.to(x.dtype)
        
        # Token-by-token Delta Rule (required for correctness)
        outputs = []
        for t in range(T):
            k_t = k[:, t, :, :]  # [B, H, K]
            v_t = v[:, t, :, :]  # [B, H, V]
            beta_t = beta[:, t, :]  # [B, H]
            g_t = g[:, t, :] if g is not None else None
            
            out_t, state = self._delta_rule_step(state, k_t, v_t, beta_t, g_t)
            outputs.append(out_t)
        
        # Stack outputs: [B, T, H, V]
        output = torch.stack(outputs, dim=1)
        
        # Project and residual
        output = output.reshape(B, T, H * V)
        output = self.o_proj(output)
        output = x + output
        
        # Diagnostics
        diagnostics = {
            'beta_mean': beta.mean().item(),
            'beta_std': beta.std().item(),
            'beta_min': beta.min().item(),
            'beta_max': beta.max().item(),
            'g_mean': g.mean().item() if g is not None else 1.0,
            'state_norm': state.norm().item(),
            'state_max': state.abs().max().item(),
        }
        
        if output_state:
            return output, state, diagnostics
        return output, None, diagnostics


# =============================================================================
# DELTA RULE VALIDATION TEST
# =============================================================================

def validate_delta_rule(verbose=True):
    """
    Test that the Delta Rule correctly suppresses redundant updates.
    
    With identical tokens, the second update should be ~zero because
    error = v - S·k ≈ v - v = 0 (after first write)
    """
    B, H, K, V = 1, 4, 16, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test layer
    class TestConfig:
        d_model = 64
        n_heads = H
        head_dim = K
        value_dim = V
    
    # Manual test of the math
    print("=" * 60)
    print("DELTA RULE VALIDATION")
    print("=" * 60)
    
    # Initialize state
    state = torch.zeros(B, H, K, V, device=device)
    
    # Create identical k, v (normalized k)
    k = torch.randn(B, H, K, device=device)
    k = F.normalize(k, p=2, dim=-1)
    v = torch.randn(B, H, V, device=device)
    
    beta = torch.tensor([1.0], device=device).expand(B, H)
    
    # First update
    prediction_1 = torch.einsum('bhkv,bhk->bhv', state, k)
    error_1 = v - prediction_1
    update_1 = torch.einsum('bhv,bhk->bhkv', error_1, k)
    state_after_1 = state + beta.unsqueeze(-1).unsqueeze(-1) * update_1
    
    if verbose:
        print(f"\n--- First Token (k, v) ---")
        print(f"  Initial state norm: {state.norm().item():.4f}")
        print(f"  Prediction norm:    {prediction_1.norm().item():.4f}")
        print(f"  Error norm:         {error_1.norm().item():.4f}")
        print(f"  Update norm:        {update_1.norm().item():.4f}")
        print(f"  State after:        {state_after_1.norm().item():.4f}")
    
    # Second update (SAME k, v)
    prediction_2 = torch.einsum('bhkv,bhk->bhv', state_after_1, k)
    error_2 = v - prediction_2
    update_2 = torch.einsum('bhv,bhk->bhkv', error_2, k)
    state_after_2 = state_after_1 + beta.unsqueeze(-1).unsqueeze(-1) * update_2
    
    if verbose:
        print(f"\n--- Second Token (SAME k, v) ---")
        print(f"  Prediction norm:    {prediction_2.norm().item():.4f}")
        print(f"  v norm:             {v.norm().item():.4f}")
        print(f"  Error norm:         {error_2.norm().item():.4f}  ← Should be ~0!")
        print(f"  Update norm:        {update_2.norm().item():.4f}  ← Should be ~0!")
        print(f"  State after:        {state_after_2.norm().item():.4f}")
    
    # Validation
    error_ratio = error_2.norm().item() / (error_1.norm().item() + 1e-8)
    state_growth = state_after_2.norm().item() / state_after_1.norm().item()
    
    print(f"\n--- Validation ---")
    print(f"  Error reduction:    {error_ratio:.6f} (should be ~0)")
    print(f"  State growth:       {state_growth:.4f} (should be ~1.0)")
    
    if error_ratio < 0.01 and state_growth < 1.1:
        print(f"\n  ✓ [PASS] TRUE DELTA RULE: Redundant information suppressed!")
        return True
    else:
        print(f"\n  ✗ [FAIL] NOT Delta Rule: State still growing with redundant data")
        return False


# =============================================================================
# NUMERICAL STABILITY TEST
# =============================================================================

def test_numerical_stability(n_tokens=100, verbose=True):
    """
    Test that state stays bounded over many tokens.
    
    With proper Delta Rule + key normalization, state should stabilize.
    """
    B, H, K, V = 1, 4, 16, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    state = torch.zeros(B, H, K, V, device=device)
    
    print("=" * 60)
    print(f"NUMERICAL STABILITY TEST ({n_tokens} tokens)")
    print("=" * 60)
    
    state_norms = []
    
    for t in range(n_tokens):
        # Random k, v each step
        k = torch.randn(B, H, K, device=device)
        k = F.normalize(k, p=2, dim=-1)
        v = torch.randn(B, H, V, device=device)
        
        # Low beta (gatekeeper style)
        beta = torch.full((B, H), 0.12, device=device)  # sigmoid(-2)
        
        # Forget gate for stability
        g = torch.full((B, H), 0.95, device=device)  # Slight decay
        
        # Delta Rule step
        prediction = torch.einsum('bhkv,bhk->bhv', state, k)
        error = v - prediction
        update = torch.einsum('bhv,bhk->bhkv', error, k)
        update = beta.unsqueeze(-1).unsqueeze(-1) * update
        state = g.unsqueeze(-1).unsqueeze(-1) * state + update
        
        state_norms.append(state.norm().item())
    
    if verbose:
        print(f"  State norm at t=0:   {state_norms[0]:.4f}")
        print(f"  State norm at t=10:  {state_norms[min(10, n_tokens-1)]:.4f}")
        print(f"  State norm at t=50:  {state_norms[min(50, n_tokens-1)]:.4f}")
        print(f"  State norm at t=99:  {state_norms[-1]:.4f}")
        print(f"  Max state norm:      {max(state_norms):.4f}")
    
    # Check for explosion
    if max(state_norms) < 100:
        print(f"\n  ✓ [PASS] State bounded: max={max(state_norms):.2f}")
        return True
    else:
        print(f"\n  ✗ [FAIL] State explosion: max={max(state_norms):.2f}")
        return False


print("GatedDeltaNetLayer loaded (TRUE DELTA RULE, PURE PYTORCH).")
print("  - Token-by-token updates for correct error correction")
print("  - Key normalization for stability")
print("  - Optional forget gate for bounded state")
print()
print("Run validate_delta_rule() to verify correctness")
print("Run test_numerical_stability() to verify bounded state")
